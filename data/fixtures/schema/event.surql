DEFINE TABLE event;

DEFINE FIELD name       ON event TYPE string;
DEFINE FIELD venue      ON event TYPE record<venue>;
DEFINE FIELD scheduled  ON event TYPE datetime;
DEFINE FIELD capacity   ON event TYPE int;
DEFINE FIELD status     ON event TYPE string;
DEFINE FIELD created    ON event TYPE datetime;
