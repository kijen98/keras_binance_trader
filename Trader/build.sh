seperator=`pwd | awk '{ split($0,arr,"/"); printf("%s\n",arr[3]);}'`
sudo docker build -t trader:$seperator .