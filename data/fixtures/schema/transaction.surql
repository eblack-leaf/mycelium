DEFINE TABLE transaction;

DEFINE FIELD amount   ON transaction TYPE float;
DEFINE FIELD type     ON transaction TYPE string;
DEFINE FIELD status   ON transaction TYPE string;
DEFINE FIELD account  ON transaction TYPE record<account>;
DEFINE FIELD created  ON transaction TYPE datetime;
