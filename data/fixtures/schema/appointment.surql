DEFINE TABLE appointment;

DEFINE FIELD patient    ON appointment TYPE record<patient>;
DEFINE FIELD doctor     ON appointment TYPE record<doctor>;
DEFINE FIELD scheduled  ON appointment TYPE datetime;
DEFINE FIELD status     ON appointment TYPE string;
DEFINE FIELD notes      ON appointment TYPE string;
