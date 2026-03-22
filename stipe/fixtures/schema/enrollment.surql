DEFINE TABLE enrollment;

DEFINE FIELD student    ON enrollment TYPE record<student>;
DEFINE FIELD course     ON enrollment TYPE record<course>;
DEFINE FIELD grade      ON enrollment TYPE float;
DEFINE FIELD completed  ON enrollment TYPE bool;
DEFINE FIELD created    ON enrollment TYPE datetime;
