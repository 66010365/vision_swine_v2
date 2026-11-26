--
-- PostgreSQL database dump
--

\restrict 7AeNN4IieVnHbAf0ik3SxvGcGTyXnflpqNoor3XZkU8JZm3v7bJWkcvf2cxkyzx

-- Dumped from database version 18.6 (Ubuntu 18.6-1.pgdg24.04+2)
-- Dumped by pg_dump version 18.6 (Ubuntu 18.6-1.pgdg24.04+2)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

-- *not* creating schema, since initdb creates it


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: cameras; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cameras (
    camera_id bigint CONSTRAINT cameras_id_not_null NOT NULL,
    house_id bigint NOT NULL,
    camera_name character varying(255) CONSTRAINT cameras_name_not_null NOT NULL,
    ip_address inet,
    resolution character varying(50),
    status character varying(20) DEFAULT 'offline'::character varying NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    camera_type character varying(20),
    description text,
    CONSTRAINT cameras_status_check CHECK (((status)::text = ANY ((ARRAY['online'::character varying, 'offline'::character varying, 'maintenance'::character varying])::text[])))
);


--
-- Name: cameras_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.cameras ALTER COLUMN camera_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.cameras_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: counting_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.counting_events (
    event_id bigint NOT NULL,
    session_id uuid NOT NULL,
    track_id integer NOT NULL,
    direction character varying(3) NOT NULL,
    frame_number integer,
    event_time timestamp with time zone NOT NULL,
    weight_estimation_id bigint,
    CONSTRAINT chk_counting_events_direction CHECK (((direction)::text = ANY ((ARRAY['IN'::character varying, 'OUT'::character varying])::text[]))),
    CONSTRAINT chk_counting_events_frame CHECK (((frame_number IS NULL) OR (frame_number >= 0))),
    CONSTRAINT chk_counting_events_track CHECK ((track_id > 0))
);


--
-- Name: counting_events_event_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.counting_events ALTER COLUMN event_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.counting_events_event_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: devices; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.devices (
    device_id bigint CONSTRAINT devices_id_not_null NOT NULL,
    device_name character varying(255) CONSTRAINT devices_name_not_null NOT NULL,
    device_type character varying(50),
    serial_number character varying(255),
    house_id bigint,
    status character varying(20) DEFAULT 'offline'::character varying NOT NULL,
    last_seen timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT devices_status_check CHECK (((status)::text = ANY ((ARRAY['online'::character varying, 'offline'::character varying, 'maintenance'::character varying])::text[])))
);


--
-- Name: devices_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.devices ALTER COLUMN device_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.devices_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: farms; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.farms (
    farm_id bigint CONSTRAINT farms_id_not_null NOT NULL,
    farm_name character varying(255) CONSTRAINT farms_name_not_null NOT NULL,
    location text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: farms_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.farms ALTER COLUMN farm_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.farms_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: houses; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.houses (
    house_id bigint CONSTRAINT houses_id_not_null NOT NULL,
    farm_id bigint NOT NULL,
    house_name character varying(255) CONSTRAINT houses_name_not_null NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    description text
);


--
-- Name: houses_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.houses ALTER COLUMN house_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.houses_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: processing_sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.processing_sessions (
    session_id uuid DEFAULT gen_random_uuid() NOT NULL,
    device_id bigint NOT NULL,
    camera_id bigint,
    source_type character varying(10) NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    ended_at timestamp with time zone,
    CONSTRAINT chk_sessions_source CHECK (((source_type)::text = ANY ((ARRAY['LIVE'::character varying, 'VIDEO'::character varying])::text[]))),
    CONSTRAINT chk_sessions_time CHECK (((ended_at IS NULL) OR (ended_at >= started_at)))
);


--
-- Name: video_recordings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.video_recordings (
    video_id bigint NOT NULL,
    session_id uuid NOT NULL,
    file_name character varying(255) NOT NULL,
    file_path text NOT NULL,
    started_at timestamp with time zone,
    ended_at timestamp with time zone,
    file_size_bytes bigint,
    status character varying(20) DEFAULT 'RECORDING'::character varying NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_video_recordings_filename CHECK ((length(TRIM(BOTH FROM file_name)) > 0)),
    CONSTRAINT chk_video_recordings_filepath CHECK ((length(TRIM(BOTH FROM file_path)) > 0)),
    CONSTRAINT chk_video_recordings_size CHECK (((file_size_bytes IS NULL) OR (file_size_bytes >= 0))),
    CONSTRAINT chk_video_recordings_status CHECK (((status)::text = ANY ((ARRAY['RECORDING'::character varying, 'READY'::character varying, 'FAILED'::character varying])::text[]))),
    CONSTRAINT chk_video_recordings_time CHECK (((ended_at IS NULL) OR ((started_at IS NOT NULL) AND (ended_at >= started_at))))
);


--
-- Name: video_recordings_video_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.video_recordings ALTER COLUMN video_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.video_recordings_video_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: weight_estimations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.weight_estimations (
    estimation_id bigint NOT NULL,
    session_id uuid NOT NULL,
    track_id integer NOT NULL,
    weight_kg numeric(7,2) NOT NULL,
    frame_number integer NOT NULL,
    estimated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_weight_estimations_frame CHECK ((frame_number >= 0)),
    CONSTRAINT chk_weight_estimations_track CHECK ((track_id > 0)),
    CONSTRAINT chk_weight_estimations_weight CHECK (((weight_kg > (0)::numeric) AND (weight_kg <> 'NaN'::numeric)))
);


--
-- Name: weight_estimations_estimation_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.weight_estimations ALTER COLUMN estimation_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.weight_estimations_estimation_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: cameras cameras_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cameras
    ADD CONSTRAINT cameras_pkey PRIMARY KEY (camera_id);


--
-- Name: counting_events counting_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.counting_events
    ADD CONSTRAINT counting_events_pkey PRIMARY KEY (event_id);


--
-- Name: devices devices_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.devices
    ADD CONSTRAINT devices_pkey PRIMARY KEY (device_id);


--
-- Name: farms farms_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.farms
    ADD CONSTRAINT farms_pkey PRIMARY KEY (farm_id);


--
-- Name: houses houses_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.houses
    ADD CONSTRAINT houses_pkey PRIMARY KEY (house_id);


--
-- Name: processing_sessions processing_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.processing_sessions
    ADD CONSTRAINT processing_sessions_pkey PRIMARY KEY (session_id);


--
-- Name: weight_estimations uq_weight_estimations_reference; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.weight_estimations
    ADD CONSTRAINT uq_weight_estimations_reference UNIQUE (estimation_id, session_id, track_id);


--
-- Name: video_recordings video_recordings_file_path_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.video_recordings
    ADD CONSTRAINT video_recordings_file_path_key UNIQUE (file_path);


--
-- Name: video_recordings video_recordings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.video_recordings
    ADD CONSTRAINT video_recordings_pkey PRIMARY KEY (video_id);


--
-- Name: weight_estimations weight_estimations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.weight_estimations
    ADD CONSTRAINT weight_estimations_pkey PRIMARY KEY (estimation_id);


--
-- Name: idx_cameras_house_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cameras_house_id ON public.cameras USING btree (house_id);


--
-- Name: idx_devices_house_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_devices_house_id ON public.devices USING btree (house_id);


--
-- Name: idx_houses_farm_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_houses_farm_id ON public.houses USING btree (farm_id);


--
-- Name: cameras cameras_house_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cameras
    ADD CONSTRAINT cameras_house_id_fkey FOREIGN KEY (house_id) REFERENCES public.houses(house_id) ON DELETE RESTRICT;


--
-- Name: devices devices_house_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.devices
    ADD CONSTRAINT devices_house_id_fkey FOREIGN KEY (house_id) REFERENCES public.houses(house_id) ON DELETE SET NULL;


--
-- Name: counting_events fk_counting_events_session; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.counting_events
    ADD CONSTRAINT fk_counting_events_session FOREIGN KEY (session_id) REFERENCES public.processing_sessions(session_id) ON DELETE RESTRICT;


--
-- Name: counting_events fk_counting_events_weight; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.counting_events
    ADD CONSTRAINT fk_counting_events_weight FOREIGN KEY (weight_estimation_id, session_id, track_id) REFERENCES public.weight_estimations(estimation_id, session_id, track_id) ON DELETE RESTRICT;


--
-- Name: processing_sessions fk_sessions_camera; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.processing_sessions
    ADD CONSTRAINT fk_sessions_camera FOREIGN KEY (camera_id) REFERENCES public.cameras(camera_id) ON DELETE RESTRICT;


--
-- Name: processing_sessions fk_sessions_device; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.processing_sessions
    ADD CONSTRAINT fk_sessions_device FOREIGN KEY (device_id) REFERENCES public.devices(device_id) ON DELETE RESTRICT;


--
-- Name: video_recordings fk_video_recordings_session; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.video_recordings
    ADD CONSTRAINT fk_video_recordings_session FOREIGN KEY (session_id) REFERENCES public.processing_sessions(session_id) ON DELETE RESTRICT;


--
-- Name: weight_estimations fk_weight_estimations_session; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.weight_estimations
    ADD CONSTRAINT fk_weight_estimations_session FOREIGN KEY (session_id) REFERENCES public.processing_sessions(session_id) ON DELETE RESTRICT;


--
-- Name: houses houses_farm_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.houses
    ADD CONSTRAINT houses_farm_id_fkey FOREIGN KEY (farm_id) REFERENCES public.farms(farm_id) ON DELETE RESTRICT;


--
-- PostgreSQL database dump complete
--

\unrestrict 7AeNN4IieVnHbAf0ik3SxvGcGTyXnflpqNoor3XZkU8JZm3v7bJWkcvf2cxkyzx

