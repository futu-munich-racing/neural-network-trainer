def train_val_test_split_session_records(
    session_records: dict,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
):

    all_train_records = list()
    all_val_records = list()
    all_test_records = list()
    for session_name in session_records.keys():
        # print(session_name, len(session_records[session_name]))
        train_records, val_records, test_records = train_val_test_split_records(
            records=session_records[session_name],
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )

        all_train_records += train_records
        all_val_records += val_records
        all_test_records += test_records

    return all_train_records, all_val_records, all_test_records


def train_val_test_split_records(
    records: list,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> (list, list, list):
    """Split given records in three parts: train, validation and test.
    We assume that the records are already in sorted based on time.
    """

    num_samples = len(records)

    # TODO: Normalise splits in a good way (or make sure that they are already)
    assert (train_split + val_split + test_split) == 1.0

    # Compute the number of samples in each split
    num_train_samples = int(num_samples * train_split)
    num_val_samples = int(num_samples * val_split)
    num_test_samples = num_samples - num_train_samples - num_val_samples

    # Split records
    train_records = records[0:num_train_samples]
    val_records = records[num_train_samples : (num_train_samples + num_val_samples)]
    test_records = records[(num_train_samples + num_val_samples) :]

    return train_records, val_records, test_records
