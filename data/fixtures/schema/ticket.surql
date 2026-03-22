DEFINE TABLE ticket;

DEFINE FIELD event     ON ticket TYPE record<event>;
DEFINE FIELD attendee  ON ticket TYPE record<attendee>;
DEFINE FIELD price     ON ticket TYPE float;
DEFINE FIELD status    ON ticket TYPE string;
DEFINE FIELD issued    ON ticket TYPE datetime;
