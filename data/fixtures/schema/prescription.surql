DEFINE TABLE prescription;

DEFINE FIELD patient  ON prescription TYPE record<patient>;
DEFINE FIELD doctor   ON prescription TYPE record<doctor>;
DEFINE FIELD drug     ON prescription TYPE string;
DEFINE FIELD dosage   ON prescription TYPE string;
DEFINE FIELD active   ON prescription TYPE bool;
DEFINE FIELD issued   ON prescription TYPE datetime;
