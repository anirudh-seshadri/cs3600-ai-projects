TOLERANCE = 0.01
from helpers import (
    EOS,
    EOS_token,
    SOS,
    SOS_token,
    Vocab,
    normalize_string,
    make_vocab,
    insert_eos_sos,
    perplexity1,
    perplexity2,
    perplexity3,
)


def test_unigram_probabilities(build_unigram_function: callable) -> None:
    """
    Test the provided build unigram function for correctness.

    This function verifies that the given build unigram function
    correctly calculates the probabilities of words based on their
    frequencies. It uses a predefined set of word probabiliities
    and asserts that the computed probabilities match the expected values.

    Parameters:
    build_unigram_function (callable): A function that takes a string
    identifiers a vocabulary, and returns the unigram probability
    of each word in the string.

    Raises:
    AssertionError: If any of the computed probabilities do not match the
    expected values.

    Prints:
    A message indicating that all tests passed if all assertions are true.
    """
    sonnet_word_probabilities = {
        "EOS": ("EOS", 0.02574291637871458),
        "SOS": ("SOS", 0.02574291637871458),
        "thou": ("thou", 0.01744989633724948),
        "beauty": ("beauty", 0.0031098825155494126),
    }
    with open("sonnets.txt", "r") as f:
        sonnets = f.read()
    sonnets = normalize_string(insert_eos_sos(sonnets))
    SONNETS_VOCAB = make_vocab([sonnets], "sonnets")

    sonnet_unigram_lm = build_unigram_function(sonnets, SONNETS_VOCAB)
    print("Beginning sonnet tests")
    for word in sonnet_word_probabilities:
        # print(sonnet_unigram_lm[SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][0])])
        assert (
            abs(
                sonnet_unigram_lm[
                    SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][0])
                ]
                - sonnet_word_probabilities[word][1]
            )
            <= TOLERANCE
        )
        print(" - Passed check: ", word)
    print("Passed sonnet tests!")

    observation_word_probabilities = {
        "EOS": ("EOS", 0.07799051466713508),
        "SOS": ("SOS", 0.07799051466713508),
        "you": ("you", 0.06021429826102231),
        "kitchen": ("kitchen", 0.005304760231863693),
    }
    with open("observations.txt", "r") as f:
        observations = f.read()
    observations = normalize_string(insert_eos_sos(observations))
    OBSERVATION_VOCAB = make_vocab([observations], "observations")

    observation_unigram_lm = build_unigram_function(observations, OBSERVATION_VOCAB)
    print("Beginning observation tests")
    for word in observation_word_probabilities:
        # print(observation_unigram_lm[OBSERVATION_VOCAB.word2token(observation_word_probabilities[word][0])])
        assert (
            abs(
                observation_unigram_lm[
                    OBSERVATION_VOCAB.word2token(
                        observation_word_probabilities[word][0]
                    )
                ]
                - observation_word_probabilities[word][1]
            )
            <= TOLERANCE
        )
        print(" - Passed check: ", word)
    print("Passed observation tests!")

    print("All tests passed!")


def test_bigram_probabilities(build_bigram_function: callable) -> None:
    """
    Test the provided build bigram function for correctness.

    This function verifies that the given build bigram function
    correctly calculates the probabilities of words based on their
    frequencies. It uses a predefined set of word probabiliities
    and asserts that the computed probabilities match the expected values.

    Parameters:
    build_bigram_function (callable): A function that takes a string
    identifiers a vocabulary, and returns the bigram probability
    of each tuple in the string.

    Raises:
    AssertionError: If any of the computed probabilities do not match the
    expected values.

    Prints:
    A message indicating that all tests passed if all assertions are true.
    """
    sonnet_word_probabilities = {
        "SOS betwixt": ("SOS", "betwixt", 0.006711409395973154),
        "SOS but": ("SOS", "but", 0.06711409395973154),
        "me EOS": ("me", "EOS", 0.17391304347826086),
        "friend EOS": ("friend", "EOS", 0.2),
        "thou art": ("thou", "art", 0.1782178217821782),
        "my joy": ("my", "joy", 0.008547008547008548),
    }
    with open("sonnets.txt", "r") as f:
        sonnets = f.read()
    sonnets = normalize_string(insert_eos_sos(sonnets))
    SONNETS_VOCAB = make_vocab([sonnets], "sonnets")

    sonnet_bigram_lm = build_bigram_function(sonnets, SONNETS_VOCAB)
    print("Beginning sonnet tests")
    for word in sonnet_word_probabilities:
        # print(sonnet_bigram_lm[SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][0])][SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][1])])
        assert (
            abs(
                sonnet_bigram_lm[
                    SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][0])
                ][SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][1])]
                - sonnet_word_probabilities[word][2]
            )
            <= TOLERANCE
        )
        print(" - Passed check: ", word)
    print("Passed sonnet tests!")

    observation_word_probabilities = {
        "SOS there": ("SOS", "there", 0.140990990990991),
        "SOS you": ("SOS", "you", 0.21396396396396397),
        "closed EOS": ("closed", "EOS", 0.6206293706293706),
        "coin EOS": ("coin", "EOS", 0.6956521739130435),
        "you see": ("you", "see", 0.7187864644107351),
        "a sofa": ("a", "sofa", 0.0013717421124828531),
    }
    with open("observations.txt", "r") as f:
        observations = f.read()
    observations = normalize_string(insert_eos_sos(observations))
    OBSERVATION_VOCAB = make_vocab([observations], "observations")

    observation_bigram_lm = build_bigram_function(observations, OBSERVATION_VOCAB)
    print("Beginning observations tests")
    for word in observation_word_probabilities:
        # print(observation_bigram_lm[OBSERVATION_VOCAB.word2token(observation_word_probabilities[word][0])][OBSERVATION_VOCAB.word2token(observation_word_probabilities[word][1])])
        assert (
            abs(
                observation_bigram_lm[
                    OBSERVATION_VOCAB.word2token(
                        observation_word_probabilities[word][0]
                    )
                ][OBSERVATION_VOCAB.word2token(observation_word_probabilities[word][1])]
                - observation_word_probabilities[word][2]
            )
            <= TOLERANCE
        )
        print(" - Passed check: ", word)
    print("Passed observations tests!")

    print("All tests passed!")


def test_trigram_probabilities(build_trigram_function: callable) -> None:
    """
    Test the provided build trigram function for correctness.

    This function verifies that the given build trigram function
    correctly calculates the probabilities of words based on their
    frequencies. It uses a predefined set of word probabiliities
    and asserts that the computed probabilities match the expected values.

    Parameters:
    build_trigram_function (callable): A function that takes a string
    identifiers a vocabulary, and returns the trigram probability
    of each tuple in the string.

    Raises:
    AssertionError: If any of the computed probabilities do not match the
    expected values.

    Prints:
    A message indicating that all tests passed if all assertions are true.
    """
    sonnet_word_probabilities = {
        "SOS thou art": ("SOS", "thou", "art", 1.0),
        "SOS how heavy": ("SOS", "how", "heavy", 0.14285714285714285),
        "so dear EOS": ("so", "dear", "EOS", 1.0),
        "of me EOS": ("of", "me", "EOS", 0.5),
        "thou wilt be": ("thou", "wilt", "be", 0.3333333333333333),
        "my love thou": ("my", "love", "thou", 0.14285714285714285),
    }

    with open("sonnets.txt", "r") as f:
        sonnets = f.read()
    sonnets = normalize_string(insert_eos_sos(sonnets))
    SONNETS_VOCAB = make_vocab([sonnets], "sonnets")

    sonnet_trigram_lm = build_trigram_function(sonnets, SONNETS_VOCAB)
    print("Beginning sonnet tests")
    for word in sonnet_word_probabilities:
        # print(sonnet_trigram_lm[SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][0])][SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][1])][SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][2])])
        assert (
            abs(
                sonnet_trigram_lm[
                    SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][0])
                ][SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][1])][
                    SONNETS_VOCAB.word2token(sonnet_word_probabilities[word][2])
                ]
                - sonnet_word_probabilities[word][3]
            )
            <= TOLERANCE
        )
        print(" - Passed check: ", word)
    print("Passed sonnet tests!")

    observation_word_probabilities = {
        "SOS there is": ("SOS", "there", "is", 1.0),
        "SOS in one": ("SOS", "in", "one", 0.6242603550295858),
        "screen door EOS": ("screen", "door", "EOS", 0.7083333333333334),
        "is closed EOS": ("is", "closed", "EOS", 1.0),
        "you see a": ("you", "see", "a", 0.7248376623376623),
        "see a sofa": ("see", "a", "sofa", 0.001702127659574468),
    }

    with open("observations.txt", "r") as f:
        observations = f.read()
    observations = normalize_string(insert_eos_sos(observations))
    OBSERVATION_VOCAB = make_vocab([observations], "observations")

    observation_trigram_lm = build_trigram_function(observations, OBSERVATION_VOCAB)
    print("Beginning observations tests")
    for word in observation_word_probabilities:
        # print(observation_trigram_lm[OBSERVATION_VOCAB.word2token(observation_word_probabilities[word][0])][OBSERVATION_VOCAB.word2token(observation_word_probabilities[word][1])][OBSERVATION_VOCAB.word2token(observation_word_probabilities[word][2])])
        assert (
            abs(
                observation_trigram_lm[
                    OBSERVATION_VOCAB.word2token(
                        observation_word_probabilities[word][0]
                    )
                ][
                    OBSERVATION_VOCAB.word2token(
                        observation_word_probabilities[word][1]
                    )
                ][
                    OBSERVATION_VOCAB.word2token(
                        observation_word_probabilities[word][2]
                    )
                ]
                - observation_word_probabilities[word][3]
            )
            <= TOLERANCE
        )
        print(" - Passed check: ", word)
    print("Passed observations tests!")

    print("All tests passed!")


def test_generate_from_unigram(
    generate_from_unigram_function: callable, build_unigram_function: callable
) -> None:

    with open("sonnets.txt", "r") as f:
        sonnets = f.read()
    sonnets = normalize_string(insert_eos_sos(sonnets))
    SONNETS_VOCAB = make_vocab([sonnets], "sonnets")

    sonnet_unigram_lm = build_unigram_function(sonnets, SONNETS_VOCAB)
    sonnet_generation_short = generate_from_unigram_function(
        SOS_token, sonnet_unigram_lm, SONNETS_VOCAB, max_length=3
    )
    sonnet_generation_long = generate_from_unigram_function(
        SOS_token, sonnet_unigram_lm, SONNETS_VOCAB, max_length=50
    )
    print(sonnet_generation_long)
    print(sonnet_generation_short)
    assert len(sonnet_generation_short) <= 3
    assert len(sonnet_generation_long) <= 50
    assert sonnet_generation_short[len(sonnet_generation_short) - 1] is EOS
    assert sonnet_generation_long[len(sonnet_generation_long) - 1] is EOS

    print("All tests passed!")


def test_generate_from_bigram(
    generate_from_bigram_function: callable, build_bigram_function: callable
) -> None:

    with open("sonnets.txt", "r") as f:
        sonnets = f.read()
    sonnets = normalize_string(insert_eos_sos(sonnets))
    SONNETS_VOCAB = make_vocab([sonnets], "sonnets")

    sonnet_bigram_lm = build_bigram_function(sonnets, SONNETS_VOCAB)
    sonnet_generation_short = generate_from_bigram_function(
        SOS_token, sonnet_bigram_lm, SONNETS_VOCAB, max_length=3
    )
    sonnet_generation_long = generate_from_bigram_function(
        SOS_token, sonnet_bigram_lm, SONNETS_VOCAB, max_length=50
    )
    print(sonnet_generation_long)
    print(sonnet_generation_short)
    assert len(sonnet_generation_short) <= 3
    assert len(sonnet_generation_long) <= 50
    assert sonnet_generation_short[len(sonnet_generation_short) - 1] is EOS
    assert sonnet_generation_long[len(sonnet_generation_long) - 1] is EOS

    print("All tests passed!")


def test_generate_from_trigram(
    generate_from_trigram_function: callable, build_trigram_function: callable
) -> None:

    with open("sonnets.txt", "r") as f:
        sonnets = f.read()
    sonnets = normalize_string(insert_eos_sos(sonnets))
    SONNETS_VOCAB = make_vocab([sonnets], "sonnets")

    sonnet_trigram_lm = build_trigram_function(sonnets, SONNETS_VOCAB)
    sonnet_generation_short = generate_from_trigram_function(
        SOS_token, sonnet_trigram_lm, SONNETS_VOCAB, max_length=3
    )
    sonnet_generation_long = generate_from_trigram_function(
        SOS_token, sonnet_trigram_lm, SONNETS_VOCAB, max_length=50
    )
    print(sonnet_generation_long)
    print(sonnet_generation_short)
    assert len(sonnet_generation_short) <= 3
    assert len(sonnet_generation_long) <= 50
    assert sonnet_generation_short[len(sonnet_generation_short) - 1] is EOS
    assert sonnet_generation_long[len(sonnet_generation_long) - 1] is EOS

    print("All tests passed!")


def test_perplexity(
    low_text: str,
    high_text: str,
    build_unigram_function: callable,
    build_bigram_function: callable,
    build_trigram_function: callable,
) -> None:

    with open("observations.txt", "r") as f:
        observations = f.read()
    observations = normalize_string(insert_eos_sos(observations))
    OBSERVATION_VOCAB = make_vocab([observations], "observations")

    observation_unigram_lm = build_unigram_function(observations, OBSERVATION_VOCAB)
    observation_bigram_lm = build_bigram_function(observations, OBSERVATION_VOCAB)
    observation_trigram_lm = build_trigram_function(observations, OBSERVATION_VOCAB)

    LOW_PERPLEXITY = 3
    HIGH_PERPLEXITY = 7

    high_seq = [
        OBSERVATION_VOCAB.word2token(w)
        for w in normalize_string(insert_eos_sos(high_text)).split()
    ]
    low_seq = [
        OBSERVATION_VOCAB.word2token(w)
        for w in normalize_string(insert_eos_sos(low_text)).split()
    ]

    assert (
        perplexity3(
            high_seq,
            observation_unigram_lm,
            observation_bigram_lm,
            observation_trigram_lm,
        )
        >= HIGH_PERPLEXITY
    )
    assert (
        perplexity3(
            low_seq,
            observation_unigram_lm,
            observation_bigram_lm,
            observation_trigram_lm,
        )
        <= LOW_PERPLEXITY
    )

    print("All tests passed!")


def test_model_perplexity(
    build_unigram_function: callable,
    build_bigram_function: callable,
    build_trigram_function: callable,
) -> None:

    # Expected perplexity
    # Training Results
    # Unigram perplexity:  3.6666260378436633
    # Bigram perplexity:  0.6778091086310991
    # Trigram perplexity:  0.39588731924520815

    # Testing Results
    # Unigram perplexity:  3.7433632389720164
    # Bigram perplexity:  0.8105437264808384
    # Trigram perplexity:  0.7156995256106654

    with open("observations.txt", "r") as f:
        observations = f.read()
    sil = normalize_string(insert_eos_sos(observations))
    sil = " ".join(sil.split()[:2000])
    sil_train = " ".join(sil.split()[:1500])
    sil_test = " ".join(sil.split()[1500:])

    SIL_VOCAB = make_vocab([sil], "sil")

    observation_unigram_lm = build_unigram_function(sil_train, SIL_VOCAB)
    observation_bigram_lm = build_bigram_function(sil_train, SIL_VOCAB)
    observation_trigram_lm = build_trigram_function(sil_train, SIL_VOCAB)

    print("Training Results:")

    token_sequence = [SIL_VOCAB.word2token(w) for w in sil_train.split()]

    unigram_train_perplexity = perplexity1(token_sequence, observation_unigram_lm)
    print("Unigram perplexity: ", unigram_train_perplexity)
    assert unigram_train_perplexity - 3.6666260378436633 <= TOLERANCE
    print("Passed unigram training perplexity")

    bigram_train_perplexity = perplexity2(
        token_sequence, observation_unigram_lm, observation_bigram_lm
    )
    print("Bigram perplexity: ", bigram_train_perplexity)
    assert bigram_train_perplexity - 0.6778091086310991 <= TOLERANCE
    print("Passed bigram training perplexity")

    trigram_train_perplexity = perplexity3(
        token_sequence,
        observation_unigram_lm,
        observation_bigram_lm,
        observation_trigram_lm,
    )
    print("Trigram perplexity: ", trigram_train_perplexity)
    assert trigram_train_perplexity - 0.39588731924520815 <= TOLERANCE
    print("Passed trigram training perplexity")

    print("Testing Results:")

    testing_token_seq = [
        SIL_VOCAB.word2token(w) for w in normalize_string(sil_test).split()
    ]

    unigram_test_perplexity = perplexity1(testing_token_seq, observation_unigram_lm)
    print("Unigram perplexity: ", unigram_test_perplexity)
    assert unigram_test_perplexity - 3.7433632389720164 <= TOLERANCE
    print("Passed unigram test perplexity")

    bigram_test_perplexity = perplexity2(
        testing_token_seq, observation_unigram_lm, observation_bigram_lm
    )
    print("Bigram perplexity: ", bigram_test_perplexity)
    assert bigram_test_perplexity - 0.8105437264808384 <= TOLERANCE
    print("Passed bigram test perplexity")

    trigram_test_perplexity = perplexity3(
        testing_token_seq,
        observation_unigram_lm,
        observation_bigram_lm,
        observation_trigram_lm,
    )
    print("Trigram perplexity: ", trigram_test_perplexity)
    assert trigram_test_perplexity - 0.7156995256106654 <= TOLERANCE
    print("Passed trigram test perplexity")

    print("All tests passed!")
