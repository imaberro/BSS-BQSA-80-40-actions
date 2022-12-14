##This file is for create a table in the database

CREATE SEQUENCE datos_ejecucion_seq;
CREATE SEQUENCE datos_iteracion_id_seq;
CREATE SEQUENCE resultado_ejecucion_id_seq;

CREATE TABLE
    datos_ejecucion
    (
        id INTEGER DEFAULT nextval('datos_ejecucion_seq'::regclass) NOT NULL,
        nombre_algoritmo CHARACTER VARYING(100),
        parametros TEXT,
        inicio TIMESTAMP(6) WITHOUT TIME ZONE,
        fin TIMESTAMP(6) WITHOUT TIME ZONE,
        estado CHARACTER VARYING(20),
        PRIMARY KEY (id)
    );

CREATE TABLE
    datos_iteracion
    (
        id BIGINT DEFAULT nextval('datos_iteracion_id_seq'::regclass) NOT NULL,
        id_ejecucion BIGINT,
        numero_iteracion INTEGER,
        fitness_mejor DOUBLE PRECISION,
        fitness_promedio DOUBLE PRECISION,
        fitness_mejor_iteracion DOUBLE PRECISION,
        parametros_iteracion TEXT,
        inicio TIMESTAMP(6) WITHOUT TIME ZONE,
        fin TIMESTAMP(6) WITHOUT TIME ZONE,
        datos_internos BYTEA,
        CONSTRAINT convergencia_pkey PRIMARY KEY (id),
        CONSTRAINT convergencia_fk1 FOREIGN KEY (id_ejecucion) REFERENCES "datos_ejecucion" ("id")
    );

CREATE TABLE
    resultado_ejecucion
    (
        id BIGINT DEFAULT nextval('resultado_ejecucion_id_seq'::regclass) NOT NULL,
        id_ejecucion BIGINT,
        fitness DOUBLE PRECISION,
        inicio TIMESTAMP(6) WITHOUT TIME ZONE,
        fin TIMESTAMP(6) WITHOUT TIME ZONE,
        mejor_solucion TEXT,
        PRIMARY KEY (id),
        CONSTRAINT resultadoejecucion_fk1 FOREIGN KEY (id_ejecucion) REFERENCES "datos_ejecucion"
        ("id")
    );
