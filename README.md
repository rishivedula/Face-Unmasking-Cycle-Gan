# Face-Unmasking-Cycle-Gan

This repo is built to model a face mask unmasking model using Pix-2-Pix model.
# Setting Up the Twitter Scraper Platform     
### Dependencies

```
python = 3.8.5
anaconda = 4.9.2
```
### 1. Setup using an Anaconda Environment

If you have not installed Anaconda, follow the instructions here (https://docs.anaconda.com/anaconda/install/index.html).

#### Once Anaconda is installed, use the following command to create a new conda environment. 
``` 
conda create -n "name"
```
To activate and deactivate the conda environment, use the following commands.
```
TO ACTIVATE
conda activate "name of the env"

TO DEACTIVATE 
conda deactivate 
```
#### Install Dependencies.
```
pip install -r package-list.txt

or 
pip3 install -r package-list.txt 

*Use these commands with or without "sudo" based on user environment conditions.
```

#### In case the Ananconda environment does not have Scipy installed.
```
sudo apt-get install python3-scipy
sudo pip install scipy

```

## Setup the Twitter Developer Account and Creating an APP

###	Twitter Developer Account
#### Follow the steps below to create the Twitter developer account, which we use to access the tweets and scrape its media objects. 
- Click https://developer.twitter.com/
- Click "Apply"
- Click "Apply for a developer account."
- Login with your Twitter Account and fill the form.
- You will receive an email upon approval of your Twitter developer account.
###	Creating an app
- Once your Twitter Developer account has been approved, you have to create an application.
- Click on your name at the top left and press get started. You will then see a tile to create an application. Press the tile, and then follow the instructions to create an application with the Twitter developer portal.
- While creating the app, you will be asked to generate tokens and API keys that we use to call the Twitter API, so  make sure you save these tokens in a text document and paste them to the "twitter_api_config.json" file. 

#### Note:-  Follow this article to see the steps for creating a Twitter developer account. 

## Setup the IBM Watson for Speech to Text Analysis.
- https://cloud.ibm.com/catalog/services/speech-to-text
- Create an account with the IBM cloud to access this API. 
- To use the free API, click on the Lite option and hit create. 
- Once your API has been created, you have to click the manage button, and there you will be able to see the API key and URL link. Copy and paste them to the "IBM_speech_to_text_config.json" file. 

#### Note: - Follow this YouTube video till "2:55" to use help with creating an IBM account for a speech-to-text setup.

## Usage

```
python twitter_scraper.py [-h] [--output_directory [OUTPUT_DIRECTORY]] [--twitterusers [TWITTERUSERS]] [--parse_tweets [PARSE_TWEETS]] [--download [DOWNLOAD]] [--analyze [ANALYZE]] [--keyword [KEYWORD]] [--num_count [NUM_COUNT]] [--text [TEXT]] [--photo [PHOTO]] [--video [VIDEO]] 

```
## Arguments
#### -h, --help	
Show this help message and exit.
#### --output_directory [OUTPUT_DIRECTORY]
Output directory where all the scraped media will be downloaded in the following format.
```
UniqueTwitterID\
	UniqueTwitterID.json
	UniqueTwitterID-source.json
	media\
		UniqueTwitterID .ext (image .jpg, etc or video)
		UniqueTwitterID .ext (audio)
		UniqueTwitterID .ext (text if any .txt)
	analysis\
              	UniqueTwitterID-analysis.json
		UniqueTwitterID-transcription.txt
		UniqueTwitterID-OCR.txt    	
		keyframes\
			UniqueTwitterID_001.jpg
			UniqueTwitterID_002.jpg
```

### User names or Search texts 
##### --twitterusers [TWITTERUSERS] OR --keyword [KEYWORD]
Specify one of the following :-
1. The keyword you want to search tweets for (e.g.:- covid19).
2. The Screen name of the user you want to search for without the "@" symbol. (e.g.:- JoeBiden from @JoeBiden, CNN from @CNN) 
#### --parse_tweets [PARSE_TWEETS]
This field is set to 1 by default which instructs the Scraper to scrape tweets. If you wish only to download or analyze the existing tweets, set this field to 0 for skipping the parsing functionality.
#### --download [DOWNLOAD] and --analyze [ANALYZE]
Both the Download and the ANALYZE fields are set to 1 by default. Acknowledge that the ANALYSE Functionality 
will NOT work if downloaded tweets are NOT available. Ideally, use these two functionalities together for the best results.

#### --num_count [NUM_COUNT]
Specify the number of tweets you wish to scrape. Also, if the number of tweets requested is significantly larger than the amount of tweets available, then the Twitter API will return the maximum amount of tweets available. Hence, you might observe a number lower than the amount requested. The 
default value has been set to 20000, which is significantly large, aiming at retrieving the maximum number of tweets that the API can return.

### Media Based Filters for Downloads
Use the below fields to filter the tweets for the desired media. Written media highly dominate Twitter, and the amount of text will be
significantly higher.  
#### --text [TEXT]
1 to download the text only
#### --photo [PHOTO]
1 to download photo only
#### --video [VIDEO]
1 to download video only


## Author
Rishi Satyanarayan Vedula <br />
[rishisat@buffalo.edu](rishisat@buffalo.edu)
