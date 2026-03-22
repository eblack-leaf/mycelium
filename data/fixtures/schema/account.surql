DEFINE TABLE account;

DEFINE FIELD name     ON account TYPE string;
DEFINE FIELD balance  ON account TYPE float;
DEFINE FIELD type     ON account TYPE string;
DEFINE FIELD owner    ON account TYPE record<customer>;
DEFINE FIELD active   ON account TYPE bool;
DEFINE FIELD created  ON account TYPE datetime;
