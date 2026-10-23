#define READ_LBF_NO_MAIN
#define CLASSICAL_BF_ONLY
#define LBF_PROGRAM_NAME "test_bf"
#include "read_lbf.c"

static void usage(const char *program) {
	fprintf(stderr,
			"Usage: %s --nonkey_csv_path NONKEYS.csv --n_test_nonkeys N "
			"--filter_file_name NAME\n"
			"Reads NAME.classical.json and computes empirical FPR on the first N "
			"non-key CSV rows (the header is skipped).\n",
			program);
}

static const char *option_value(int argc, char **argv, const char *option) {
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], option) == 0) {
			if (i + 1 >= argc || argv[i + 1][0] == '-') {
				fprintf(stderr, "test_bf: missing value for %s\n", option);
				exit(EXIT_FAILURE);
			}
			return argv[i + 1];
		}
	}
	return NULL;
}

static int classical_filter_contains(const JsonParser *parser, size_t filter,
									 const double *features, size_t feature_count) {
	double raw_m = number_value(parser, required_field(parser, filter, "m"));
	double n = number_value(parser, required_field(parser, filter, "n"));
	if (raw_m < 1.0 || raw_m > (double)INT64_MAX || floor(raw_m) != raw_m || n <= 0.0)
		fail("invalid classical Bloom filter dimensions");
	size_t m = (size_t)raw_m;
	size_t bitmap = required_field(parser, filter, "bitmap");
	if (array_length(parser, bitmap) != m)
		fail("classical Bloom filter bitmap length does not match m");
	size_t hash_count = (size_t)ceil(((double)m / n) * log(2.0));
	if (hash_count == 0) fail("invalid classical Bloom filter hash count");

	/* The Python test reads CSV values as float32 before hashing their bytes. */
	float *key = malloc(feature_count * sizeof(*key));
	if (key == NULL) fail("out of memory hashing key");
	for (size_t i = 0; i < feature_count; i++) key[i] = (float)features[i];

	for (size_t seed = 0; seed < hash_count; seed++) {
		int32_t hash = murmur3_32((const unsigned char *)key,
								  feature_count * sizeof(*key), (uint32_t)seed);
		int64_t remainder = (int64_t)hash % (int64_t)m;
		if (remainder < 0) remainder += (int64_t)m;
		size_t bit = array_get(parser, bitmap, (size_t)remainder);
		if (token_equals(parser, bit, "false") || token_equals(parser, bit, "0")) {
			free(key);
			return 0;
		}
		if (!token_equals(parser, bit, "true") && !token_equals(parser, bit, "1"))
			fail("invalid value in classical Bloom filter bitmap");
	}
	free(key);
	return 1;
}

static size_t count_classical_positives(const char *path, const JsonParser *parser,
										size_t filter, size_t feature_count,
										double *features, size_t wanted) {
	FILE *cases = fopen(path, "r");
	if (cases == NULL) {
		fprintf(stderr, "test_bf: cannot open '%s': %s\n", path, strerror(errno));
		exit(EXIT_FAILURE);
	}
	char *line = NULL;
	size_t line_capacity = 0;
	ssize_t line_length;
	if (getline(&line, &line_capacity, cases) < 0)
		fail("CSV input is empty; expected a header row");
	size_t line_number = 1, rows = 0, positives = 0;
	while (rows < wanted && (line_length = getline(&line, &line_capacity, cases)) >= 0) {
		line_number++;
		char *nonspace = line;
		while (*nonspace == ' ' || *nonspace == '\t' || *nonspace == '\r' || *nonspace == '\n')
			nonspace++;
		if (line_length == 0 || *nonspace == '\0') continue;
		parse_csv_features(line, features, feature_count, line_number);
		positives += classical_filter_contains(parser, filter, features, feature_count) ? 1 : 0;
		rows++;
	}
	if (ferror(cases)) fail("error reading CSV cases");
	if (rows < wanted) {
		fprintf(stderr, "test_bf: '%s' has only %zu rows, %zu requested\n", path, rows, wanted);
		exit(EXIT_FAILURE);
	}
	free(line);
	fclose(cases);
	return positives;
}

static size_t csv_feature_count(const char *path) {
	FILE *cases = fopen(path, "r");
	if (cases == NULL) {
		fprintf(stderr, "test_bf: cannot open '%s': %s\n", path, strerror(errno));
		exit(EXIT_FAILURE);
	}
	char *header = NULL;
	size_t capacity = 0;
	ssize_t length = getline(&header, &capacity, cases);
	if (length < 0) fail("CSV input is empty; expected a header row");
	size_t columns = 1;
	for (ssize_t i = 0; i < length; i++)
		if (header[i] == ',') columns++;
	free(header);
	fclose(cases);
	return columns;
}

int main(int argc, char **argv) {
	if (argc == 2 && strcmp(argv[1], "--help") == 0) {
		usage(argv[0]);
		return EXIT_SUCCESS;
	}
	const char *nonkeys_path = option_value(argc, argv, "--nonkey_csv_path");
	const char *count_text = option_value(argc, argv, "--n_test_nonkeys");
	const char *filter_name = option_value(argc, argv, "--filter_file_name");
	if (nonkeys_path == NULL || count_text == NULL || filter_name == NULL) {
		usage(argv[0]);
		return EXIT_FAILURE;
	}
	size_t nonkey_total = parse_count(count_text, "N_TEST_NONKEYS");

	const char *suffix = strrchr(filter_name, '.') != NULL &&
						 strcmp(strrchr(filter_name, '.'), ".json") == 0
						 ? "" : ".classical.json";
	size_t model_path_length = strlen(filter_name) + strlen(suffix) + 1;
	char *model_path = malloc(model_path_length);
	if (model_path == NULL) fail("out of memory creating model path");
	snprintf(model_path, model_path_length, "%s%s", filter_name, suffix);

	size_t json_length;
	char *json = read_file(model_path, &json_length);
	JsonParser parser = {json, json_length, 0, NULL, 0, 0};
	size_t root = parse_value(&parser);
	skip_whitespace(&parser);
	if (parser.position != parser.length || parser.tokens[root].type != JSON_OBJECT)
		fail("input must contain one JSON object");
	size_t filter_class = required_field(&parser, root, "filter_class");
	if (!token_equals(&parser, filter_class, "ClassicalBloomFilterImpl"))
		fail("only ClassicalBloomFilterImpl JSON filters are supported");
	size_t filter_size = (size_t)number_value(
		&parser, required_field(&parser, root, "m"));
	size_t feature_count = csv_feature_count(nonkeys_path);
	double *features = malloc(feature_count * sizeof(*features));
	if (features == NULL) fail("out of memory allocating features");
	size_t false_positives = count_classical_positives(
		nonkeys_path, &parser, root, feature_count, features, nonkey_total);
	printf("False positive rate: %.6f.\n",
		   (double)false_positives / (double)nonkey_total);
	printf("Size of the classical Bloom filter: %zu bits.\n", filter_size);

	free(features);
	free(parser.tokens);
	free(json);
	free(model_path);
	return EXIT_SUCCESS;
}