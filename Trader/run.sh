seperator=`pwd | awk '{ split($0,arr,"/"); printf("%s\n",arr[3]);}'`
sudo docker run -d -v "/AI_Project/$seperator:/app/" --restart=always --name $seperator trader:$seperator