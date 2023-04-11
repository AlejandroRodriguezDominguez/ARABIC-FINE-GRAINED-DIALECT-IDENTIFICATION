================================================================
                    MADAR SHARED TASK 2019
            Arabic Fine-Grained Dialect Identification

   The Fourth Workshop for Arabic Natural Language Processing
                         (WANLP 2019)

                    Fourth Release of Data
                         18 March 2019

=================================================================
Updates in Fourth Release of Data, 18 March 2019
=================================================================

In this release of the data, we provide ALL of the training and
development data for both SUBTASK ***1*** and SUBTASK ***2***.
There will be no additional releases except for the test files.

This README file is present in the zipped folder:
MADAR-SHARED-TASK-fourth-release-18Mar2019.zip

=================================================================
MADAR-Shared-Task-Subtask-1
=================================================================

Description of Data:
--------------------

The provided train and development files correspond to Corpus-26
and Corpus-6 in Salameh et al. (2018).

* Corpus-26 train set (MADAR-Corpus26-train.tsv) consists of
1,600 sentences in 26 versions (MSA and 25 city dialects) =
1,600*(1 + 25) = 41,600 sentences, with their corresponding labels.

* Corpus-26 dev set (MADAR-Corpus26-dev.tsv) consists of 200
sentences in 26 versions  (MSA and 25 city dialects) =
200*(1 + 25) =  5,200 sentences, with their corresponding labels.

* Corpus-6 train set (MADAR-Corpus6-train.tsv) consists of 9,000
sentences in 6 versions (MSA and 5 city dialects) = 9,000*(1 + 5)
=  54,000 sentences, with their corresponding labels.

* Corpus-6 dev set (MADAR-Corpus6-dev.tsv) consists of 1,000
sentences in 6 versions (MSA and 5 city dialects) =
1,000*(1 + 5) =  6,000 sentences, with their corresponding labels.

The MADAR Shared Task subtask 1 test set (MADAR-Corpus26-test.tsv)
is *NOT* included in this release. It will be made available
later, during the evaluation phase.  The test set will consists of
200 sentences in 26 versions (MSA and 25 city dialects) =
200*(1 + 25) =  5,200 sentences.

*** A note on MADAR Corpus: Bouamor et al. (2018) describe
Corpus-5 and Corpus-25, which are the same as Corpus-6 and
Corpus-26, respectively, except for the addition of MSA data.

Shared Task Metrics and Restrictions:
-------------------------------------

The performance of submitted systems will be evaluated on
MADAR-Corpus26-test.tsv which will be made available during the
evaluation phase. MADAR-Corpus6-train.tsv and
MADAR-Corpus6-dev.tsv are provided to aid building the models.
Participants are welcome to use both of these files for training
purposes.

The training data from MADAR-Shared-Task-Subtask-2 is allowed.
External manually labelled data sets are *NOT* allowed.
However, the use of publicly available unlabelled data is allowed. 

IMPORTANT: Participants are NOT allowed to use
MADAR-Corpus26-dev.tsv for training purposes. Participants must
report the performance of their best system on
MADAR-Corpus26-dev.tsv in their Shared Task system description
paper.

The labels of dialects for the cities and MSA are defined in
Salameh et al. (2018): ALE, ALG, ALX, AMM, ASW, BAG, BAS, BEI,
BEN, CAI, DAM, DOH, FES, JED, JER, KHA, MOS, MUS, RAB, RIY, SAL,
SAN, SFX, TRI, TUN and MSA.

EXAMPLE.GOLD and EXAMPLE.PRED are example gold and prediction
files which can be used with the MADAR-DID-Scorer.py
provided (description below).

MADAR-Corpus-Lexicon-License.txt contains the license for using
this data.


=================================================================
MADAR-Shared-Task-Subtask-2
=================================================================

Description of Data:
--------------------

The goal of Subtask 2 is to predict countries of Twitter users by
using information about tweets posted by the Twitter users. The
provided train and development data contain two files each (for
a total of four files):

(1) MADAR-Twitter-Subtask-2.TRAIN.user-label.tsv
(2) MADAR-Twitter-Subtask-2.TRAIN.user-tweets-features.tsv
(3) MADAR-Twitter-Subtask-2.DEV.user-label.tsv
(4) MADAR-Twitter-Subtask-2.DEV.user-tweets-features.tsv

### The User-Label files

The *.user-label.tsv files contain Twitter users and corresponding
labels of their countries. The country labels are: Algeria,
Bahrain, Djibouti, Egypt, Iraq, Jordan, Kuwait, Lebanon, Libya,
Mauritania, Morocco, Oman, Palestine, Qatar, Saudi_Arabia, Somalia,
Sudan, Syria, Tunisia, United_Arab_Emirates, Yemen.

### The User-Tweets-Features files

The *.user-tweets-features.tsv files contain information about
tweets posted by users in the *.user-label.tsv files. Due to
restrictions by Twitter, we do not provide the tweets but provide
tweet IDs and information about tweets. We provide a script
(MADAR-Obtain-Tweets.py) to obtain the tweets -- more
information about this script is below.

In the *.user-tweets-features.tsv files, we provide the following
information:

- The first column (#1 User) contains username of Twitter user
from the *.user-label.tsv file.

- The second column (#2 Tweet ID) contains ID of tweet posted by
the Twitter user.

- The third column (#3 Automatically Detected Language by Twitter)
contains language of the tweet as detected by Twitter.

- The fourth column (#4 Features ALE,ALG,ALX,AMM,ASW,BAG,BAS,BEI,
BEN,CAI,DAM,DOH,FES,JED,JER,KHA,MOS,MSA,MUS,RAB,RIY,SAL,SAN,
SFX,TRI,TUN) contains comma-separated probability scores of 25
city dialects and MSA for each tweet. The probability scores
are obtained by running the best model described in Salameh
et al. (2018).  The tweets were preprocessed to exclude all
non-Arabic characters. If a tweet becomes empty after excluding
non-Arabic characters, the cell contains the value <NIL>.

- The fifth column (#5 Label: Country of User) contains label of
country for the Twitter user.

### File Statistics

Below, we present more details about the contents of the train
and development files.

(1) MADAR-Twitter-Subtask-2.TRAIN.user-label.tsv contains 2180
users and corresponding labels of their countries.

(2) MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.tsv
contains information about up to 100 tweets for each of the 2180
profiles in the MADAR-Twitter-Subtask-2.TRAIN.user-label.tsv
file (a total of 217,592 tweets).

(3) MADAR-Twitter-Subtask-2.DEV.user-label.tsv contains 300
users and corresponding labels of their countries.

(4) MADAR-Twitter-Subtask-2.DEV.user-tweet-features.tsv
contains information about up to 100 tweets for each of the 300
profiles in the MADAR-Twitter-Subtask-2.DEV.user-label.tsv
file (a total of 29,869 tweets).

The MADAR Shared Task Subtask 2 test set files
(MADAR-Twitter-Subtask-2.TEST.user-label.tsv and
MADAR-Twitter-Subtask-2.TEST.user-tweets-features.tsv)
are *NOT* included in this release. They will be made available
later, during the evaluation phase. The test set will consist
of 500 users and information about up to 100 tweets for each
of the users.

Shared Task Metrics and Restrictions:
-------------------------------------

The performance of submitted systems will be evaluated on
predictions of country labels for Twitter users in
MADAR-Twitter-Subtask-2.TEST.user-label.tsv.

IMPORTANT: Participants are NOT allowed to use
MADAR-Twitter-Subtask-2.DEV.user-label.tsv and
MADAR-Twitter-Subtask-2.DEV.user-tweet-features.tsv
for training purposes. Participants must report the performance
of their best system on MADAR-Twitter-Subtask-2.DEV.user-label.tsv
in their Shared Task system description paper.

IMPORTANT: Participants can only use the ***text*** of the tweets
obtained through (MADAR-Obtain-Tweets.py) and the specific
information about the tweets provided in
MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.tsv.
Participants are NOT allowed to use additional tweets, nor
are they allowed to use outside information about the Twitter User.
Specifically -- participants should not use meta
data from Twitter about the users or the tweets, e.g.,
geo-location data.

The training data from MADAR-Shared-Task-Subtask-1 is allowed.
External manually labelled data sets are *NOT* allowed.
However, the use of publicly available unlabelled data is allowed.

EXAMPLE.GOLD and EXAMPLE.PRED are example gold and prediction
files which can be used with the MADAR-DID-Scorer.py
provided (description below).

MADAR-Twitter-Corpus-License.txt contains the license for using
this data.

MADAR-Obtain-Tweets.py:
-----------------------------

MADAR-Obtain-Tweets.py is a python script that will take in
a *.user-tweets-features.tsv file and append a column containing
actual tweet texts corresponding to tweet IDs in the file.

VERY-IMPORTANT (1):  Please make sure to have unicodecsv and tweepy
libraries installed. Make the following calls in terminal:

     pip install unicodecsv
     pip install tweepy

VERY-IMPORTANT (2): Use of this script will require a Twitter
developer's account and corresponding authentication credentials.
The authentication credentials have to be provided in lines
81-84 of the code.

Creating a developer's account and obtaining credentials:
To create a Twitter developer's account, login to Twitter and
go to https://developer.twitter.com/en/apply-for-access.html.
After applying for the developer's account, you will have to wait for the
application to be approved. Twitter may contact you for additional
information about your usage. Please make sure to check your email.

Once the developer's account is created, you will need to create an app.
To create an app, go to apps under the dropdown at your username (upper
right corner). You can now click "create an app."

After an app is created, go to your app details and go to the
"keys and tokens" tab. You will be able to generate following
authentication credentials required for the script:

(1) consumer key
(2) consumer secret
(3) access token
(4) access secret


Usage of the script:

    python3 MADAR-Obtain-Tweets.py  <input_file> <output_file>

    The input file must be in the same format as the
    MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.tsv
    or MADAR-Twitter-Subtask-2.DEV.user-label.tsv.

    The output file is the name of the file where the obtained tweets
    will be appended. If any tweet is unavailable, it will write
    <UNAVAILABLE> for that tweet.

Example usage:

    python3 MADAR-Obtain-Tweets.py  MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.tsv MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.full-tweets.tsv

Running the following command will produce the file
MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.full-tweets.tsv that
contains obtained tweets appended in the last column along with
the information from
MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.tsv


=================================================================
/MADAR-DID-Scorer.py
=================================================================

The scoring script (MADAR-DID-Scorer.py) is provided at the
root directory of the released data.  MADAR-DID-Scorer.py
is a python script that will take in two text files containing
true labels and predicted labels and will output accuracy,
F1 score, precision and recall. (Note that the official metric is
F1 score).  The scoring script is used for both
MADAR-Shared-Task-Subtask-1 and MADAR-Shared-Task-Subtask-2.

Please make sure to have sklearn library installed.

Usage of the scorer:

    python3 MADAR-DID-Scorer.py  <gold-file> <pred-file>

    For verbose mode:

    python3 MADAR-DID-Scorer.py  <gold-file> <pred-file> -verbose

In the MADAR-Shared-Task-Subtask-1 and
MADAR-Shared-Task-Subtask-2 directories, there are example
gold and prediction files. If they are used with the scorer,
they should produce the following results:

python3 MADAR-DID-Scorer.py MADAR-Shared-Task-Subtask-1/EXAMPLE.GOLD MADAR-Shared-Task-Subtask-1/EXAMPLE.PRED

OVERALL SCORES:
MACRO AVERAGE PRECISION SCORE: 80.56 %
MACRO AVERAGE RECALL SCORE: 75.00 %
MACRO AVERAGE F1 SCORE: 73.89 %
OVERALL ACCURACY: 75.00 %

python3 MADAR-DID-Scorer.py MADAR-Shared-Task-Subtask-2/EXAMPLE.GOLD MADAR-Shared-Task-Subtask-2/EXAMPLE.PRED

OVERALL SCORES:
MACRO AVERAGE PRECISION SCORE: 75.32 %
MACRO AVERAGE RECALL SCORE: 78.33 %
MACRO AVERAGE F1 SCORE: 75.40 %
OVERALL ACCURACY: 77.50 %

=================================================================
References
=================================================================

[1] Salameh, Mohammad, Houda Bouamor, and Nizar Habash. "Fine-
Grained Arabic Dialect Identification." Proceedings of the
27th International Conference on Computational Linguistics.
Santa Fe, New Mexico, USA, 2018.
http://aclweb.org/anthology/C18-1113

[2] Bouamor, Houda, Nizar Habash, Mohammad Salameh, Wajdi
Zaghouani, Owen Rambow, Dana Abdulrahim, Ossama Obeid, Salam
Khalifa, Fadhl Eryani, Alexander Erdmann and Kemal Oflazer.
"The MADAR Arabic Dialect Corpus and Lexicon."  Proceedings
of the Eleventh International Conference on Language Resources
and Evaluation (LREC 2018). Miyazaki, Japan, 2018.
http://www.lrec-conf.org/proceedings/lrec2018/pdf/351.pdf

================================================================
Copyright (c) 2019 Carnegie Mellon University Qatar
and New York University Abu Dhabi. All rights reserved.
================================================================
