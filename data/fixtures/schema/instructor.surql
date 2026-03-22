DEFINE TABLE instructor;

DEFINE FIELD name        ON instructor TYPE string;
DEFINE FIELD email       ON instructor TYPE string;
DEFINE FIELD department  ON instructor TYPE record<department>;
DEFINE FIELD active      ON instructor TYPE bool;
DEFINE FIELD created     ON instructor TYPE datetime;
