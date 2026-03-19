DEFINE TABLE post;

DEFINE FIELD title    ON post TYPE string;
DEFINE FIELD body     ON post TYPE string;
DEFINE FIELD author   ON post TYPE record<user>;
DEFINE FIELD tags     ON post TYPE array<string>;
DEFINE FIELD created  ON post TYPE datetime;
