DEFINE TABLE patient;

DEFINE FIELD name     ON patient TYPE string;
DEFINE FIELD email    ON patient TYPE string;
DEFINE FIELD phone    ON patient TYPE string;
DEFINE FIELD dob      ON patient TYPE datetime;
DEFINE FIELD insured  ON patient TYPE bool;
DEFINE FIELD created  ON patient TYPE datetime;
