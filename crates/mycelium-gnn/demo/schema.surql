-- Demo schema: blog platform with users, posts, comments

DEFINE TABLE users SCHEMAFULL;
DEFINE FIELD name ON users TYPE string;
DEFINE FIELD age ON users TYPE int;
DEFINE FIELD email ON users TYPE string;
DEFINE FIELD created ON users TYPE datetime;

DEFINE TABLE posts SCHEMAFULL;
DEFINE FIELD title ON posts TYPE string;
DEFINE FIELD content ON posts TYPE string;
DEFINE FIELD author ON posts TYPE record<users>;
DEFINE FIELD created ON posts TYPE datetime;

DEFINE TABLE comments SCHEMAFULL;
DEFINE FIELD text ON comments TYPE string;
DEFINE FIELD author ON comments TYPE record<users>;
DEFINE FIELD post ON comments TYPE record<posts>;
DEFINE FIELD created ON comments TYPE datetime;
