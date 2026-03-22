DEFINE TABLE expense;

DEFINE FIELD amount    ON expense TYPE float;
DEFINE FIELD description ON expense TYPE string;
DEFINE FIELD employee  ON expense TYPE record<employee>;
DEFINE FIELD approved  ON expense TYPE bool;
DEFINE FIELD created   ON expense TYPE datetime;
