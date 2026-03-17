-- Expanded demo schema: social platform + e-commerce

DEFINE TABLE users SCHEMAFULL;
DEFINE FIELD name ON users TYPE string;
DEFINE FIELD age ON users TYPE int;
DEFINE FIELD email ON users TYPE string;
DEFINE FIELD bio ON users TYPE string;
DEFINE FIELD active ON users TYPE bool;
DEFINE FIELD score ON users TYPE float;
DEFINE FIELD created ON users TYPE datetime;

DEFINE TABLE posts SCHEMAFULL;
DEFINE FIELD title ON posts TYPE string;
DEFINE FIELD content ON posts TYPE string;
DEFINE FIELD author ON posts TYPE record<users>;
DEFINE FIELD published ON posts TYPE bool;
DEFINE FIELD views ON posts TYPE int;
DEFINE FIELD rating ON posts TYPE float;
DEFINE FIELD created ON posts TYPE datetime;

DEFINE TABLE comments SCHEMAFULL;
DEFINE FIELD text ON comments TYPE string;
DEFINE FIELD author ON comments TYPE record<users>;
DEFINE FIELD post ON comments TYPE record<posts>;
DEFINE FIELD likes ON comments TYPE int;
DEFINE FIELD created ON comments TYPE datetime;

DEFINE TABLE tags SCHEMAFULL;
DEFINE FIELD name ON tags TYPE string;
DEFINE FIELD color ON tags TYPE string;

DEFINE TABLE categories SCHEMAFULL;
DEFINE FIELD name ON categories TYPE string;
DEFINE FIELD description ON categories TYPE string;
DEFINE FIELD parent ON categories TYPE option<record<categories>>;

DEFINE TABLE likes SCHEMAFULL;
DEFINE FIELD user ON likes TYPE record<users>;
DEFINE FIELD post ON likes TYPE record<posts>;
DEFINE FIELD created ON likes TYPE datetime;

DEFINE TABLE follows SCHEMAFULL;
DEFINE FIELD follower ON follows TYPE record<users>;
DEFINE FIELD following ON follows TYPE record<users>;
DEFINE FIELD created ON follows TYPE datetime;

DEFINE TABLE messages SCHEMAFULL;
DEFINE FIELD sender ON messages TYPE record<users>;
DEFINE FIELD recipient ON messages TYPE record<users>;
DEFINE FIELD body ON messages TYPE string;
DEFINE FIELD read ON messages TYPE bool;
DEFINE FIELD created ON messages TYPE datetime;

DEFINE TABLE products SCHEMAFULL;
DEFINE FIELD name ON products TYPE string;
DEFINE FIELD price ON products TYPE float;
DEFINE FIELD stock ON products TYPE int;
DEFINE FIELD description ON products TYPE string;
DEFINE FIELD category ON products TYPE record<categories>;
DEFINE FIELD rating ON products TYPE float;
DEFINE FIELD created ON products TYPE datetime;

DEFINE TABLE orders SCHEMAFULL;
DEFINE FIELD customer ON orders TYPE record<users>;
DEFINE FIELD total ON orders TYPE float;
DEFINE FIELD status ON orders TYPE string;
DEFINE FIELD quantity ON orders TYPE int;
DEFINE FIELD created ON orders TYPE datetime;

DEFINE TABLE reviews SCHEMAFULL;
DEFINE FIELD product ON reviews TYPE record<products>;
DEFINE FIELD author ON reviews TYPE record<users>;
DEFINE FIELD rating ON reviews TYPE int;
DEFINE FIELD body ON reviews TYPE string;
DEFINE FIELD created ON reviews TYPE datetime;
