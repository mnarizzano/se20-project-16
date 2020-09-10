==== ITA-PREREQ Training Data - PRELEARN @ EVALITA 2020 ====

This folder contains the training data for the Prerequisite RElation LEARNing (PRELEARN) shared task at EVALITA 2020. 

For each of the four domains (Data Mining, Geometry, Physics, Precalculus) you will find a  `.csv` training file consisting of pairs of target and prerequisite concepts (A, B) labelled as follows:
	- 1 if B is a prerequisite of A;
	- 0 in all other cases.

The Wikipedia page of each concept included in the training files can be found in the `ITA_prereq-pages.xml` file. Each Wikipedia page is introduced by a `<doc>` element (with *id* and *url*) containing the title and the text of the corresponding page.

-- STATISTICS --

The following are the quantity of positive and negative concept pairs in the trainig sets of the four domains and overall. 
Note that the test sets will be balanced.

+--------------+--------------+---------------+---------------+
|DOMAIN        |TRAIN SIZE    |POSITIVE PAIRS |NEGATIVE PAIRS |
+--------------+--------------+---------------+---------------+
|Data Mining   |          424 |           109 |           315 | 
|Geometry      |         1548 |           332 |          1216 | 
|Physics       |         1905 |           315 |          1905 |
|Precalculus   |         1308 |           408 |          1308 | 
+-------------------------------------------------------------+
|All Domains   |         5908 |          1164 |          4744 | 
+--------------+--------------+---------------+---------------+

-- PARSING XML FILE --

You can parse the `ITA_prereq-pages.xml` file using your favourite approach. We suggest to parse the xml file using `xml.etree.ElementTree` Python module as follows:

```
import xml.etree.ElementTree as ET

tree = ET.parse('ITA_prereq-pages.xml')
root= tree.getroot()
for content in root.iter('doc'):
	print('Page id: ', content.get('id'))
	print('Page url: ', content.get('url'))
	print('Page title: ', content.title)
	print('Page text: ', content.text)
```

-- TERMS AND CONDITIONS --
All data, including annotations, is provided under the CC-BY-NA 4.0 license (see https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

-- CONTACTS --

For further information contact the organizers: prelearn.evalita2020@gmail.com
