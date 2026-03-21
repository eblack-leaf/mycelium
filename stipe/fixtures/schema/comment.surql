DEFINE TABLE comment;

DEFINE FIELD body    ON comment TYPE string;
DEFINE FIELD author  ON comment TYPE record<user>;
DEFINE FIELD task    ON comment TYPE record<task>;
DEFINE FIELD created ON comment TYPE datetime;
