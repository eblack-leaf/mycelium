DEFINE TABLE course;

DEFINE FIELD name        ON course TYPE string;
DEFINE FIELD credits     ON course TYPE int;
DEFINE FIELD capacity    ON course TYPE int;
DEFINE FIELD active      ON course TYPE bool;
DEFINE FIELD instructor  ON course TYPE record<instructor>;
DEFINE FIELD created     ON course TYPE datetime;
