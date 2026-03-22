DEFINE TABLE invoice;

DEFINE FIELD amount   ON invoice TYPE float;
DEFINE FIELD paid     ON invoice TYPE bool;
DEFINE FIELD order    ON invoice TYPE record<order>;
DEFINE FIELD issued   ON invoice TYPE datetime;
