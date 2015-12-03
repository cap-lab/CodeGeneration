package hopes.cic.xml;

import static javax.xml.XMLConstants.W3C_XML_SCHEMA_NS_URI;

import java.io.File;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBElement;
import javax.xml.bind.Unmarshaller;
import javax.xml.bind.ValidationEvent;
import javax.xml.bind.ValidationEventHandler;
import javax.xml.bind.ValidationEventLocator;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;

public class JAXBTest {
	
	public final static String CONTEXT_PATH = "hopes.cic.xml";
	public final static String XSD_FILE_NAME = "xsd/CICAlgorithm.xsd";
	
	
	public static void main(String[] args) throws Exception {
		JAXBContext jc = JAXBContext.newInstance(CONTEXT_PATH);
		System.out.println(jc.toString());
		Unmarshaller u = jc.createUnmarshaller();  

		SchemaFactory sf = SchemaFactory.newInstance(W3C_XML_SCHEMA_NS_URI);
		try {
			Schema schema = sf.newSchema(new File(XSD_FILE_NAME));
			u.setSchema(schema);
			u.setEventHandler(
					new ValidationEventHandler() {
						// allow unmarshalling to continue even if there are errors
						public boolean handleEvent(ValidationEvent ve) {
							// ignore warnings
							if (ve.getSeverity() != ValidationEvent.WARNING) {
								ValidationEventLocator vel = ve.getLocator();
								System.out.println("Line:Col[" + vel.getLineNumber() +
										":" + vel.getColumnNumber() +
										"]:" + ve.getMessage());
							}
							return true;
						}
					}
			);
		} catch (org.xml.sax.SAXException se) {
			System.out.println("Unable to validate due to following error.");
			se.printStackTrace();
		}

		JAXBElement<?> poElement = 
			(JAXBElement<?>)u.unmarshal( new File( "./sample/putto_home/resources/types/type1.type" ) );
		CICAlgorithmType algorithm = (CICAlgorithmType)poElement.getValue();

		// display the shipping address
		System.out.println("name = " + algorithm.getTasks());
	}
}
