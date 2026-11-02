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

static int classical_filter_contains(const DecodedBloomFilter *filter,
									 const double *features, size_t feature_count,
									 float *key) {
	for (size_t i = 0; i < feature_count; i++) key[i] = (float)features[i];

	for (size_t seed = 0; seed < filter->hash_count; seed++) {
		int32_t hash = murmur3_32((const unsigned char *)key,
								  feature_count * sizeof(*key), (uint32_t)seed);
		int64_t remainder = (int64_t)hash % (int64_t)filter->m;
		if (remainder < 0) remainder += (int64_t)filter->m;
		if (!filter->bits[(size_t)remainder]) return 0;
	}
	return 1;
}

static size_t count_classical_positives(const char *path,
										const DecodedBloomFilter *filter,
										size_t feature_count, double *features,
										float *query_key, size_t wanted,
										QueryTiming *query_timing) {
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
		uint64_t query_start_ns = monotonic_time_ns();
		int contains = classical_filter_contains(filter, features,
												 feature_count, query_key);
		query_timing->elapsed_ns += monotonic_time_ns() - query_start_ns;
		query_timing->query_count++;
		positives += contains ? 1 : 0;
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
	DecodedBloomFilter filter = decode_bloom_filter(&parser, root);
	size_t feature_count = csv_feature_count(nonkeys_path);
	double *features = malloc(feature_count * sizeof(*features));
	if (features == NULL) fail("out of memory allocating features");
	float *query_key = malloc(feature_count * sizeof(*query_key));
	if (query_key == NULL) fail("out of memory allocating query key");
	free(parser.tokens);
	free(json);
	QueryTiming query_timing = {0, 0};
	size_t false_positives = count_classical_positives(
		nonkeys_path, &filter, feature_count, features, query_key,
		nonkey_total, &query_timing);
	printf("False positive rate: %.6f.\n",
		   (double)false_positives / (double)nonkey_total);
	printf("Size of the classical Bloom filter: %zu bits.\n", filter.m);
	printf("QUERY_TIME_NS=%llu\n", (unsigned long long)query_timing.elapsed_ns);
	printf("QUERY_COUNT=%zu\n", query_timing.query_count);

	free(features);
	free(query_key);
	free_decoded_bloom_filter(&filter);
	free(model_path);
	return EXIT_SUCCESS;
}