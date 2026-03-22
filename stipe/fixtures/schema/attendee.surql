DEFINE TABLE attendee;

DEFINE FIELD name    ON attendee TYPE string;
DEFINE FIELD email   ON attendee TYPE string;
DEFINE FIELD phone   ON attendee TYPE string;
DEFINE FIELD event   ON attendee TYPE record<event>;
DEFINE FIELD created ON attendee TYPE datetime;
