FROM python:3.8
#Labels as key value pair
LABEL Maintainer="sumanth4@illinois.edu"


# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /Users/sumababu1/sarcasm_config
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow
RUN pip3 install -r requirements.txt 
RUN pip3 install numpy --upgrade
RUN pip3 install requests
COPY . .


#to COPY the remote file at working directory in container
COPY sarcasm_analysis.py ./
# Now the structure looks like this '/usr/app/src/test.py'


#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

CMD [ "python", "./sarcasm_analysis.py"]

