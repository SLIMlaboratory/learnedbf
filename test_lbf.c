#define READ_LBF_NO_MAIN
#define LBF_PROGRAM_NAME "test_lbf"
#include "read_lbf.c"

static void usage(const char *program) {
	fprintf(stderr,
			"Usage: %s --nonkey_csv_path NONKEYS.csv --n_test_nonkeys N "
			"--filter_file_name NAME\n"
			"Reads NAME.learned.json and computes empirical FPR on the first N "
			"non-key CSV rows (the header is skipped).\n",
			program);
}

static const char *option_value(int argc, char **argv, const char *option) {
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], option) == 0) {
			if (i + 1 >= argc || argv[i + 1][0] == '-') {
				fprintf(stderr, "test_lbf: missing value for %s\n", option);
				exit(EXIT_FAILURE);
			}
			return argv[i + 1];
		}
	}
	return NULL;
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
						 ? "" : ".learned.json";
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

	size_t classifier = required_field(&parser, root, "classifier");
	size_t class_name = required_field(&parser, root, "classifier_class");
	if (!token_equals(&parser, class_name, "ScoredLinearSVC"))
		fail("only LBF exports using ScoredLinearSVC are supported");
	size_t coef = required_field(&parser, classifier, "coef_");
	size_t coef_row = array_get(&parser, coef, 0);
	size_t feature_count = array_length(&parser, coef_row);
	size_t n_features = (size_t)number_value(
		&parser, required_field(&parser, classifier, "n_features_in_"));
	if (feature_count == 0 || feature_count != n_features)
		fail("invalid serialized classifier feature count");

	double *features = malloc(feature_count * sizeof(*features));
	if (features == NULL) fail("out of memory allocating features");
	size_t intercept = required_field(&parser, classifier, "intercept_");
	size_t classes = required_field(&parser, classifier, "classes_");
	size_t false_positives = count_positives(
		nonkeys_path, &parser, root, coef_row, intercept, classes,
		feature_count, features, nonkey_total);

	double classifier_size = number_value(
		&parser, required_field(&parser, classifier, "float_size")) *
		(double)(feature_count + 1);
	double backup_size = 0.0;
	size_t backup = object_get(&parser, root, "backup_filter");
	if (backup != SIZE_MAX && !token_equals(&parser, backup, "null"))
		backup_size = number_value(&parser, required_field(&parser, backup, "m"));

	printf("False positive rate: %.6f.\n",
		   (double)false_positives / (double)nonkey_total);
	printf("Size of the learned Bloom filter: %.0f bits.\n",
		   classifier_size + backup_size);

	free(features);
	free(parser.tokens);
	free(json);
	free(model_path);
	return EXIT_SUCCESS;
}