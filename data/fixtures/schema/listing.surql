DEFINE TABLE listing;

DEFINE FIELD property  ON listing TYPE record<property>;
DEFINE FIELD agent     ON listing TYPE record<agent>;
DEFINE FIELD price     ON listing TYPE float;
DEFINE FIELD status    ON listing TYPE string;
DEFINE FIELD created   ON listing TYPE datetime;
