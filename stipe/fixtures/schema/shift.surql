DEFINE TABLE shift;

DEFINE FIELD employee ON shift TYPE record<employee>;
DEFINE FIELD start    ON shift TYPE datetime;
DEFINE FIELD end      ON shift TYPE datetime;
DEFINE FIELD location ON shift TYPE string;
DEFINE FIELD paid     ON shift TYPE bool;
