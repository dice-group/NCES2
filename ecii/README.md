- Run `python generate_config_ecii.py ` to prepare configuration files

- To start concept learning, run `java -Xms2g -Xmx8g -Xss1g -jar ecii_v1.0.0.jar -b kb/`

- Run `python parse_ecii_output.py ` to parse the output and save the results such as f_measure and runtime

kb is the name of the knowledge base, e.g., carcinogenesis, vicodi, etc.