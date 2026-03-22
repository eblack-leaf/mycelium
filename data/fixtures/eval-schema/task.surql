DEFINE TABLE task;

DEFINE FIELD title       ON task TYPE string;
DEFINE FIELD description ON task TYPE string;
DEFINE FIELD status      ON task TYPE string;
DEFINE FIELD priority    ON task TYPE int;
DEFINE FIELD assignee    ON task TYPE record<user>;
DEFINE FIELD project     ON task TYPE record<project>;
DEFINE FIELD due_date    ON task TYPE datetime;
DEFINE FIELD created     ON task TYPE datetime;
