DEFINE TABLE department;

DEFINE FIELD name    ON department TYPE string;
DEFINE FIELD budget  ON department TYPE float;
DEFINE FIELD head    ON department TYPE record<employee>;
DEFINE FIELD created ON department TYPE datetime;
