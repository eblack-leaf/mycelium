DEFINE TABLE budget;

DEFINE FIELD name        ON budget TYPE string;
DEFINE FIELD amount      ON budget TYPE float;
DEFINE FIELD spent       ON budget TYPE float;
DEFINE FIELD department  ON budget TYPE record<department>;
DEFINE FIELD active      ON budget TYPE bool;
DEFINE FIELD created     ON budget TYPE datetime;
