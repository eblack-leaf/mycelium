DEFINE TABLE shipment;

DEFINE FIELD order      ON shipment TYPE record<order>;
DEFINE FIELD carrier    ON shipment TYPE string;
DEFINE FIELD status     ON shipment TYPE string;
DEFINE FIELD weight     ON shipment TYPE float;
DEFINE FIELD estimated  ON shipment TYPE datetime;
DEFINE FIELD created    ON shipment TYPE datetime;
