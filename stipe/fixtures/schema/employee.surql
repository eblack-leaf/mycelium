DEFINE TABLE employee;

DEFINE FIELD name        ON employee TYPE string;
DEFINE FIELD email       ON employee TYPE string;
DEFINE FIELD salary      ON employee TYPE float;
DEFINE FIELD hired       ON employee TYPE datetime;
DEFINE FIELD active      ON employee TYPE bool;
DEFINE FIELD department  ON employee TYPE record<department>;
DEFINE FIELD role        ON employee TYPE record<role>;
