# Hangouts Reader
Hangouts Reader is a command line Python script that allows to read and display the conversations in a Google Hangouts logfile.

## How to download the logfile?
Google gives you a nice interface to download your data that is stored in their cloud. This does not only work for Hangouts but for a lot more services. You find it on [Google Takeout](https://www.google.com/takeout/). Go there and choose "Hangouts" and create the archive. After a short period of time you can download it via the given link into some folder `$FOLDER`. Extract the archive and you should find a file `$FOLDER/$YOUR_USERNAME@gmail.com-takeout/Hangouts/Hangouts.json`. Remember this path because we need it in a second.

## How to use the script
You can handle the script easily. Go to the path were you have cloned this git repository to and call the help of the script

    python hangouts.py --help

If you see some information about the script together with its usage information everything works fine.

Then you have to tell the script where your `Hangouts.json` file is located at. Remembering the path we found out above use

    python hangouts.py $FOLDER/$YOUR_USERNAME@gmail.com-takeout/Hangouts/Hangouts.json

If everything works out you will see a list of all conversations between you and your communication partners. A possible output could be

    [hangouts.py] conversation id: UgyUz5R_Xb2NN0EWZs54AaA4AQ, participants: Roberto Laurance, Fabian Mueller 
    [hangouts.py] conversation id: UgzpubtSJwYEm3wMzVZ4AaA4AQ, participants: Luis Mueller, Udo Maier, Fabian Mueller
    [hangouts.py] conversation id: UgwB3uEDgZsHba4ia4F4AaA4AQ, participants: Luis Mueller, Manuel Schorsch, Fabian Mueller

This allows you to choose which conversation you want to show. Use one of the identifiers, let us say the second one (`UgzpubtSJwYEm3wMzVZ4AaA4AQ`), to show the conversation. You can do this by calling

    python hangouts.py $FOLDER/$YOUR_USERNAME@gmail.com-takeout/Hangouts/Hangouts.json -c UgzpubtSJwYEm3wMzVZ4AaA4AQ

You have then a detailed overview over the complete history.

## You found an error?
Let me know if there is something not working as expected and I will try to fix it. Please see the [issues](https://bitbucket.org/dotcs/hangouts-log-reader/issues) page.
## Known Issues

* At the moment Hangouts Reader is only working with Python up to 2.7. Python 3 support will be included in future releases. Please be patient.

## Remark
Google® is a registered trademark of Google Inc.

Please note that this is a private project. Google® is not associated in any way with this project.
