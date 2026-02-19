DROP TABLE IF EXISTS coord_data;
DROP TABLE IF EXISTS ec_data;

CREATE TABLE coord_data (
    coord_id INTEGER PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    elev REAL,
    igbp TEXT
);

CREATE TABLE ec_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coord_id INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    {vars},
    FOREIGN KEY(coord_id) REFERENCES coord_data(coord_id)
);