DEFINE TABLE review;

DEFINE FIELD rating   ON review TYPE int;
DEFINE FIELD body     ON review TYPE string;
DEFINE FIELD product  ON review TYPE record<product>;
DEFINE FIELD author   ON review TYPE record<customer>;
DEFINE FIELD created  ON review TYPE datetime;
