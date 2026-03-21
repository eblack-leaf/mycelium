DEFINE TABLE project;

DEFINE FIELD name        ON project TYPE string;
DEFINE FIELD description ON project TYPE string;
DEFINE FIELD status      ON project TYPE string;
DEFINE FIELD owner       ON project TYPE record<user>;
DEFINE FIELD created     ON project TYPE datetime;
