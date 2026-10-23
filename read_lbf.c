#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#ifndef LBF_PROGRAM_NAME
#define LBF_PROGRAM_NAME "read_lbf"
#endif

/*
 * Build with: gcc -std=c11 -O2 read_lbf.c -lm -o read_lbf
 * Usage:      ./read_lbf exported_lbf.json data/keys.csv data/nonkeys.csv N_KEYS N_NONKEYS
 * The input JSON must be produced by LBF.export() using ScoredLinearSVC.
 * The first CSV row is treated as a header; subsequent rows must contain
 * exactly one numeric feature per model input feature.
 */

/* Small, purpose-built JSON tokenizer. Tokens refer to slices of the input. */
typedef enum { JSON_OBJECT, JSON_ARRAY, JSON_STRING, JSON_PRIMITIVE } JsonType;

typedef struct {
	JsonType type;
	size_t start;
	size_t end;
	size_t next;
} JsonToken;

typedef struct {
	const char *text;
	size_t length;
	size_t position;
	JsonToken *tokens;
	size_t count;
	size_t capacity;
} JsonParser;

static void fail(const char *message) {
	fprintf(stderr, "%s: %s\n", LBF_PROGRAM_NAME, message);
	exit(EXIT_FAILURE);
}

static void skip_whitespace(JsonParser *parser) {
	while (parser->position < parser->length &&
		   (parser->text[parser->position] == ' ' ||
			parser->text[parser->position] == '\t' ||
			parser->text[parser->position] == '\r' ||
			parser->text[parser->position] == '\n')) {
		parser->position++;
	}
}

static size_t add_token(JsonParser *parser, JsonType type, size_t start) {
	if (parser->count == parser->capacity) {
		size_t new_capacity = parser->capacity == 0 ? 256 : parser->capacity * 2;
		JsonToken *new_tokens = realloc(parser->tokens,
										new_capacity * sizeof(*new_tokens));
		if (new_tokens == NULL) fail("out of memory parsing JSON");
		parser->tokens = new_tokens;
		parser->capacity = new_capacity;
	}
	size_t index = parser->count++;
	parser->tokens[index] = (JsonToken){type, start, start, index + 1};
	return index;
}

static size_t parse_value(JsonParser *parser);

static size_t parse_container(JsonParser *parser, JsonType type, char closing) {
	size_t index = add_token(parser, type, parser->position++);
	skip_whitespace(parser);
	if (parser->position < parser->length && parser->text[parser->position] == closing) {
		parser->tokens[index].end = ++parser->position;
		parser->tokens[index].next = parser->count;
		return index;
	}

	while (parser->position < parser->length) {
		if (type == JSON_OBJECT) {
			if (parser->text[parser->position] != '"') fail("expected JSON object key");
			parse_value(parser);
			skip_whitespace(parser);
			if (parser->position >= parser->length || parser->text[parser->position++] != ':')
				fail("expected ':' after JSON object key");
			skip_whitespace(parser);
		}
		parse_value(parser);
		skip_whitespace(parser);
		if (parser->position >= parser->length) fail("unterminated JSON container");
		char delimiter = parser->text[parser->position++];
		if (delimiter == closing) break;
		if (delimiter != ',') fail("expected ',' between JSON values");
		skip_whitespace(parser);
	}

	parser->tokens[index].end = parser->position;
	parser->tokens[index].next = parser->count;
	return index;
}

static size_t parse_value(JsonParser *parser) {
	skip_whitespace(parser);
	if (parser->position >= parser->length) fail("unexpected end of JSON input");
	char c = parser->text[parser->position];
	if (c == '{') return parse_container(parser, JSON_OBJECT, '}');
	if (c == '[') return parse_container(parser, JSON_ARRAY, ']');

	if (c == '"') {
		size_t index = add_token(parser, JSON_STRING, parser->position++);
		while (parser->position < parser->length) {
			char current = parser->text[parser->position++];
			if (current == '\\') {
				if (parser->position < parser->length) parser->position++;
			} else if (current == '"') {
				parser->tokens[index].end = parser->position;
				parser->tokens[index].next = parser->count;
				return index;
			}
		}
		fail("unterminated JSON string");
	}

	size_t index = add_token(parser, JSON_PRIMITIVE, parser->position);
	while (parser->position < parser->length) {
		c = parser->text[parser->position];
		if (c == ',' || c == ']' || c == '}' || c == ' ' || c == '\t' ||
			c == '\r' || c == '\n') break;
		parser->position++;
	}
	if (parser->position == parser->tokens[index].start) fail("invalid JSON value");
	parser->tokens[index].end = parser->position;
	parser->tokens[index].next = parser->count;
	return index;
}

static int token_equals(const JsonParser *parser, size_t index, const char *text) {
	JsonToken token = parser->tokens[index];
	size_t length = token.end - token.start;
	if (token.type == JSON_STRING) {
		token.start++;
		length -= 2;
	}
	return strlen(text) == length && memcmp(parser->text + token.start, text, length) == 0;
}

static size_t object_get(const JsonParser *parser, size_t object, const char *key) {
	JsonToken container = parser->tokens[object];
	if (container.type != JSON_OBJECT) return SIZE_MAX;
	size_t index = object + 1;
	while (index < container.next) {
		size_t value = index + 1;
		if (token_equals(parser, index, key)) return value;
		index = parser->tokens[value].next;
	}
	return SIZE_MAX;
}

static size_t array_get(const JsonParser *parser, size_t array, size_t wanted) {
	JsonToken container = parser->tokens[array];
	if (container.type != JSON_ARRAY) return SIZE_MAX;
	size_t index = array + 1;
	while (index < container.next && wanted > 0) {
		index = parser->tokens[index].next;
		wanted--;
	}
	return index < container.next ? index : SIZE_MAX;
}

static size_t required_field(const JsonParser *parser, size_t object,
							 const char *name) {
	size_t field = object_get(parser, object, name);
	if (field == SIZE_MAX) {
		fprintf(stderr, "read_lbf: missing required JSON field '%s'\n", name);
		exit(EXIT_FAILURE);
	}
	return field;
}

static double number_value(const JsonParser *parser, size_t index) {
	if (index == SIZE_MAX || parser->tokens[index].type != JSON_PRIMITIVE)
		fail("expected a JSON number");
	char *end = NULL;
	errno = 0;
	double value = strtod(parser->text + parser->tokens[index].start, &end);
	if (errno != 0 || end != parser->text + parser->tokens[index].end || !isfinite(value))
		fail("invalid or non-finite JSON number");
	return value;
}

static size_t array_length(const JsonParser *parser, size_t index) {
	if (index == SIZE_MAX || parser->tokens[index].type != JSON_ARRAY)
		fail("expected a JSON array");
	size_t length = 0;
	for (size_t item = index + 1; item < parser->tokens[index].next;
		 item = parser->tokens[item].next) length++;
	return length;
}

/*
 * Match mmh3.hash(key, seed=seed, signed=True): seed is the full unsigned
 * 32-bit initial hash state, not an iteration count or an XOR mask.
 */
static int32_t murmur3_32(const unsigned char *data, size_t length, uint32_t seed) {
	const uint32_t c1 = 0xcc9e2d51U, c2 = 0x1b873593U;
	uint32_t hash = seed;
	size_t blocks = length / 4;
	for (size_t i = 0; i < blocks; i++) {
		const unsigned char *block = data + i * 4;
		uint32_t k = (uint32_t)block[0] |
					 ((uint32_t)block[1] << 8) |
					 ((uint32_t)block[2] << 16) |
					 ((uint32_t)block[3] << 24);
		k *= c1; k = (k << 15) | (k >> 17); k *= c2;
		hash ^= k; hash = (hash << 13) | (hash >> 19);
		hash = hash * 5U + 0xe6546b64U;
	}
	const unsigned char *tail = data + blocks * 4;
	uint32_t k = 0;
	switch (length & 3U) {
		case 3: k ^= (uint32_t)tail[2] << 16; /* fall through */
		case 2: k ^= (uint32_t)tail[1] << 8;  /* fall through */
		case 1: k ^= tail[0]; k *= c1; k = (k << 15) | (k >> 17); k *= c2; hash ^= k;
	}
	hash ^= (uint32_t)length;
	hash ^= hash >> 16; hash *= 0x85ebca6bU;
	hash ^= hash >> 13; hash *= 0xc2b2ae35U;
	hash ^= hash >> 16;
	/* Convert the same 32 hash bits to mmh3.hash's default signed result. */
	if (hash <= INT32_MAX) return (int32_t)hash;
	return -(int32_t)(UINT32_MAX - hash) - 1;
}

#ifndef CLASSICAL_BF_ONLY
static int backup_filter_contains(const JsonParser *parser, size_t filter,
								  const double *features, size_t feature_count) {
	size_t m = (size_t)number_value(parser, required_field(parser, filter, "m"));
	double n = number_value(parser, required_field(parser, filter, "n"));
	size_t bitmap = required_field(parser, filter, "bitmap");
	if (m == 0 || m > INT64_MAX || n <= 0 || array_length(parser, bitmap) != m)
		fail("invalid backup Bloom filter dimensions");
	size_t hash_count = (size_t)ceil(((double)m / n) * log(2.0));

	/* build_lbf.py loads float32 data, so Python hashes the float32 bytes. */
	float *key = malloc(feature_count * sizeof(*key));
	if (key == NULL) fail("out of memory hashing key");
	for (size_t i = 0; i < feature_count; i++) key[i] = (float)features[i];

	for (size_t seed = 0; seed < hash_count; seed++) {
		int32_t hash = murmur3_32((const unsigned char *)key,
									  feature_count * sizeof(*key), (uint32_t)seed);
		/* Python's mmh3.hash(...) % m uses signed, nonnegative modulo. */
		int64_t remainder = (int64_t)hash % (int64_t)m;
		if (remainder < 0) remainder += (int64_t)m;
		size_t bit = (size_t)remainder;
		size_t bit_token = array_get(parser, bitmap, bit);
		if (token_equals(parser, bit_token, "false") || token_equals(parser, bit_token, "0")) {
			free(key);
			return 0;
		}
		if (!token_equals(parser, bit_token, "true") && !token_equals(parser, bit_token, "1"))
			fail("invalid value in backup Bloom filter bitmap");
	}
	free(key);
	return 1;
}
#endif

static char *read_file(const char *path, size_t *length) {
	FILE *file = fopen(path, "rb");
	if (file == NULL) {
		fprintf(stderr, "%s: cannot open '%s': %s\n", LBF_PROGRAM_NAME, path, strerror(errno));
		exit(EXIT_FAILURE);
	}
	if (fseek(file, 0, SEEK_END) != 0) fail("cannot seek JSON file");
	long size = ftell(file);
	if (size < 0 || fseek(file, 0, SEEK_SET) != 0) fail("cannot read JSON file size");
	char *data = malloc((size_t)size + 1);
	if (data == NULL) fail("out of memory reading JSON file");
	if (fread(data, 1, (size_t)size, file) != (size_t)size) fail("failed to read JSON file");
	fclose(file);
	data[size] = '\0';
	*length = (size_t)size;
	return data;
}

static size_t parse_csv_features(char *line, double *features, size_t expected,
								 size_t line_number) {
	size_t count = 0;
	char *cursor = line;
	while (1) {
		while (*cursor == ' ' || *cursor == '\t') cursor++;
		if (*cursor == '\0' || *cursor == '\r' || *cursor == '\n') break;
		if (count == expected) {
			fprintf(stderr, "%s: too many columns on CSV line %zu\n", LBF_PROGRAM_NAME, line_number);
			exit(EXIT_FAILURE);
		}

		int quoted = (*cursor == '"');
		if (quoted) cursor++;
		char *end = NULL;
		errno = 0;
		double value = strtod(cursor, &end);
		if (errno != 0 || end == cursor || !isfinite(value)) {
			fprintf(stderr, "%s: invalid numeric value on CSV line %zu, column %zu\n",
					LBF_PROGRAM_NAME,
					line_number, count + 1);
			exit(EXIT_FAILURE);
		}
		cursor = end;
		while (*cursor == ' ' || *cursor == '\t') cursor++;
		if (quoted) {
			if (*cursor != '"') {
				fprintf(stderr, "%s: malformed quoted value on CSV line %zu\n", LBF_PROGRAM_NAME, line_number);
				exit(EXIT_FAILURE);
			}
			cursor++;
			while (*cursor == ' ' || *cursor == '\t') cursor++;
		}
		features[count++] = value;

		if (*cursor == ',') {
			cursor++;
			continue;
		}
		if (*cursor == '\r') cursor++;
		if (*cursor == '\n') cursor++;
		if (*cursor != '\0') {
			fprintf(stderr, "%s: unexpected character on CSV line %zu\n", LBF_PROGRAM_NAME, line_number);
			exit(EXIT_FAILURE);
		}
		break;
	}

	if (count != expected) {
		fprintf(stderr, "%s: expected %zu columns but found %zu on CSV line %zu\n",
				LBF_PROGRAM_NAME, expected, count, line_number);
		exit(EXIT_FAILURE);
	}
	return count;
}

#ifndef CLASSICAL_BF_ONLY
static int predict_case(const JsonParser *parser, size_t root, size_t coef_row,
					size_t intercept, size_t classes,
					size_t feature_count, const double *features, double *score_out) {
	double decision = 0.0;
	for (size_t i = 0; i < feature_count; i++)
		decision += features[i] * number_value(parser, array_get(parser, coef_row, i));
	decision += number_value(parser, array_get(parser, intercept, 0));
	if (token_equals(parser, array_get(parser, classes, 0), "false") ||
		token_equals(parser, array_get(parser, classes, 0), "0"))
		decision = -decision;

	/* ScoredLinearSVC.predict_score applies the sigmoid to -decision. */
	double score = decision >= 0.0 ? exp(-decision) / (1.0 + exp(-decision))
								   : 1.0 / (1.0 + exp(decision));
	int predicted = score > number_value(parser, required_field(parser, root, "threshold"));
	size_t backup = object_get(parser, root, "backup_filter");
	if (!predicted && backup != SIZE_MAX && !token_equals(parser, backup, "null"))
		predicted = backup_filter_contains(parser, backup, features, feature_count);
	*score_out = score;
	return predicted;
}
#endif

#ifndef READ_LBF_NO_MAIN
static void read_lbf_usage(const char *program) {
	fprintf(stderr, "Usage: %s MODEL.json KEYS.csv NONKEYS.csv N_KEYS N_NONKEYS\n"
					"Reads a learnedbf.LBF export with a ScoredLinearSVC, checks that the "
					"first N_KEYS keys are predicted positive and prints the false positive "
					"ratio on the first N_NONKEYS non-keys (CSV header rows are skipped).\n",
			program);
}
#endif

static size_t parse_count(const char *text, const char *name) {
	char *end = NULL;
	errno = 0;
	unsigned long long value = strtoull(text, &end, 10);
	if (errno != 0 || end == text || *end != '\0' || text[0] == '-' || value == 0) {
		fprintf(stderr, "%s: %s must be a positive integer\n", LBF_PROGRAM_NAME, name);
		exit(EXIT_FAILURE);
	}
	return (size_t)value;
}

/* Reads exactly wanted rows; returns how many were predicted positive. */
#ifndef CLASSICAL_BF_ONLY
static size_t count_positives(const char *path, const JsonParser *parser, size_t root,
							  size_t coef_row, size_t intercept, size_t classes,
							  size_t feature_count, double *features, size_t wanted) {
	FILE *cases = fopen(path, "r");
	if (cases == NULL) {
		fprintf(stderr, "%s: cannot open '%s': %s\n", LBF_PROGRAM_NAME, path, strerror(errno));
		exit(EXIT_FAILURE);
	}
	char *line = NULL;
	size_t line_capacity = 0;
	ssize_t line_length;
	if (getline(&line, &line_capacity, cases) < 0) fail("CSV input is empty; expected a header row");
	size_t line_number = 1, rows = 0, positives = 0;
	while (rows < wanted && (line_length = getline(&line, &line_capacity, cases)) >= 0) {
		line_number++;
		char *nonspace = line;
		while (*nonspace == ' ' || *nonspace == '\t' || *nonspace == '\r' || *nonspace == '\n')
			nonspace++;
		if (line_length == 0 || *nonspace == '\0') continue;
		parse_csv_features(line, features, feature_count, line_number);
		double score;
		positives += predict_case(parser, root, coef_row, intercept,
								  classes, feature_count, features, &score) ? 1 : 0;
		rows++;
	}
	if (ferror(cases)) fail("error reading CSV cases");
	if (rows < wanted) {
		fprintf(stderr, "read_lbf: '%s' has only %zu rows, %zu requested\n", path, rows, wanted);
		exit(EXIT_FAILURE);
	}
	free(line);
	fclose(cases);
	return positives;
}
#endif

#ifndef READ_LBF_NO_MAIN
int main(int argc, char **argv) {
	if (argc != 6) { read_lbf_usage(argv[0]); return EXIT_FAILURE; }
	size_t key_total = parse_count(argv[4], "N_KEYS");
	size_t nonkey_total = parse_count(argv[5], "N_NONKEYS");

	size_t length;
	char *json = read_file(argv[1], &length);
	JsonParser parser = {json, length, 0, NULL, 0, 0};
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
	size_t n_features = (size_t)number_value(&parser,
						 required_field(&parser, classifier, "n_features_in_"));
	if (feature_count != n_features) fail("invalid serialized classifier feature count");

	double *features = malloc(feature_count * sizeof(*features));
	if (features == NULL) fail("out of memory allocating features");
	size_t intercept = required_field(&parser, classifier, "intercept_");
	size_t classes = required_field(&parser, classifier, "classes_");
	size_t key_hits = count_positives(argv[2], &parser, root, coef_row, intercept,
									  classes, feature_count, features, key_total);
	if (key_hits != key_total) {
		fprintf(stderr, "read_lbf: %zu of %zu keys were not predicted as keys\n",
				key_total - key_hits, key_total);
		return EXIT_FAILURE;
	}
	size_t false_positives = count_positives(argv[3], &parser, root, coef_row, intercept,
											 classes, feature_count, features, nonkey_total);
	printf("keys: %zu, all predicted positive\n", key_total);
	printf("non-keys: %zu, false positives: %zu, FPR: %.17g\n", nonkey_total,
		   false_positives, (double)false_positives / (double)nonkey_total);
	free(features);
	free(parser.tokens);
	free(json);
	return EXIT_SUCCESS;
}
#endif
