## Usage
To create and populate the database, make sure that all sql files and the format_table.sh file within the database folder is located in the lion directory of your VM. Then in your terminal run the command:

```
sh ./format_tables.sh
```

To run the application on a web browser, make sure your directory is within the project files and simply execute:

```
export FLASK_APP=app.py
flask run
# then browse to http://127.0.0.1:5000/
```

## Screenshots
To use: Decide wether you would like to see data from municipalities in a county or a specific municipality and hit the sumbit button.

![maingui](https://github.com/TCNJ-degoodj/cab-project-9/blob/main/images/Screen%20Shot%202023-04-24%20at%2010.25.05%20PM.png)
![countGUI](https://github.com/TCNJ-degoodj/cab-project-9/blob/main/images/Screen%20Shot%202023-05-01%20at%204.50.30%20PM.png)
![muniGUI](https://github.com/TCNJ-degoodj/cab-project-9/blob/main/images/Screenshot%20at%202023-05-01%2018-16-37.png)

