DEFINE TABLE order;

DEFINE FIELD status    ON order TYPE string;
DEFINE FIELD total     ON order TYPE float;
DEFINE FIELD customer  ON order TYPE record<customer>;
DEFINE FIELD created   ON order TYPE datetime;
