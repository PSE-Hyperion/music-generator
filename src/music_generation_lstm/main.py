import cli



# TODO: SELECT SEQUENCE LENGTH DEPENDING ON MEAN AMOUNT OF EVENTS OF SCORES IN DATASET (i.e s1 has 10 event, s2 has 12 events, s3 has 11 events -> chose SEQUENCE_LENGTH as least third of mean(10,11,12))


if __name__ == "__main__":
    cli.start_session()
