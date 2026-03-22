DEFINE TABLE product;

DEFINE FIELD name      ON product TYPE string;
DEFINE FIELD price     ON product TYPE float;
DEFINE FIELD stock     ON product TYPE int;
DEFINE FIELD sku       ON product TYPE string;
DEFINE FIELD active    ON product TYPE bool;
DEFINE FIELD category  ON product TYPE record<category>;
DEFINE FIELD created   ON product TYPE datetime;
