package hopes.cic.xml;

import static javax.xml.XMLConstants.W3C_XML_SCHEMA_NS_URI;
import hopes.cic.exception.CICXMLErrorCode;
import hopes.cic.exception.CICXMLException;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBElement;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;
import javax.xml.bind.ValidationEvent;
import javax.xml.bind.ValidationEventHandler;
import javax.xml.bind.ValidationEventLocator;
import javax.xml.transform.stream.StreamSource;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;

import org.w3c.dom.ls.LSInput;
import org.w3c.dom.ls.LSResourceResolver;
import org.xml.sax.SAXException;

import com.sun.org.apache.xerces.internal.dom.DOMInputImpl;
import com.sun.xml.bind.marshaller.DataWriter;

/**
 * 
 * @author long21s
 *
 * @param <T> : wrapper class of "S"
 */
public abstract class ResourceLoader<S> {
	public final static String SCHEMA_DIR = "/xsd/";
	public final static String CIC_MODEL_PACKAGE = "hopes.cic.xml";
	private final static String SCHEMA_PREFIX = "./";
	private final static Logger logger = Logger.getLogger(ResourceLoader.class.getName());

	protected static ObjectFactory OBJECT_FACTORY = new ObjectFactory();
	private static JAXBContext JAXB_CONTEXT; // for marshaller, unmarshaller creation
	private static SchemaFactory SCHEMA_FACTORY; // for schema validation
	static { // no need to get for each resource type
		try {
			JAXB_CONTEXT = JAXBContext.newInstance(CIC_MODEL_PACKAGE);
			if (logger.isLoggable(Level.INFO)) {
				logger.info("JAXB Context : " + JAXB_CONTEXT);
			}
		} catch(JAXBException e) {
			logger.log(Level.WARNING, "JAXB Context init error", e);
		}

		SCHEMA_FACTORY = SchemaFactory.newInstance(W3C_XML_SCHEMA_NS_URI);
		SCHEMA_FACTORY.setResourceResolver(new LSResourceResolver() {
			@Override
			public LSInput resolveResource(String type,
					String namespaceURI, String publicId, String systemId,
					String baseURI) {
				String xsdFile;
				if (systemId.startsWith(SCHEMA_PREFIX)) {
					xsdFile = getXSDFilePath(systemId.substring(SCHEMA_PREFIX.length()));
				} else {
					xsdFile = systemId;
				}
				if (logger.isLoggable(Level.INFO)) {
					logger.info("type - " + type + ", publicId - " + publicId + ", systemId - " + 
							xsdFile + ", baseURI - " + baseURI);
				}
				InputStream is = ResourceLoader.class.getResourceAsStream(xsdFile);
				if (is == null) {
					logger.log(Level.WARNING, "xsd file[" + xsdFile +"] open error");
					return null;
				}
				return new DOMInputImpl(publicId, xsdFile, baseURI, is, "UTF-8");
			}
		}
		);
	}

	private static String getXSDFilePath(String xsdFile) {
		return SCHEMA_DIR + xsdFile; 
	}
	
	// initilize unmarhsaller
	private void initUnmarshaller(String xsdFile, Unmarshaller unmarshaller) 
	throws JAXBException {
		String xsdPath = getXSDFilePath(xsdFile);
		InputStream is = ResourceLoader.class.getResourceAsStream(xsdPath);
		if (is == null) {
			logger.warning("get schema[" +SCHEMA_DIR +  xsdPath + "] error");
			return;
		}
		try {
			Schema schema = SCHEMA_FACTORY.newSchema(new StreamSource(is));
			unmarshaller.setSchema(schema);
		} catch (SAXException e) { // just ignore schema validation
			logger.log(Level.WARNING, "xsd file[" + xsdPath + "] new schema error", e);
		}

		unmarshaller.setEventHandler(
				new ValidationEventHandler() {
					// allow unmarshalling to continue even if there are errors
					public boolean handleEvent(ValidationEvent ve) {
						// ignore warnings
						if (ve.getSeverity() != ValidationEvent.WARNING) {
							ValidationEventLocator vel = ve.getLocator();
							logger.warning("Line:Col[" + vel.getLineNumber() +
									":" + vel.getColumnNumber() +
									"]:" + ve.getMessage());
							return false;
						}
						return true;
					}
				}
		);
	}

	private static Marshaller MARSHALLER;
	private Marshaller getMarshaller() throws JAXBException {
		if (MARSHALLER != null)
			return MARSHALLER;

		MARSHALLER = JAXB_CONTEXT.createMarshaller();
		return MARSHALLER;
	}

	// map is needed beacause each resource has a different schema validator
	private static Map<String, Unmarshaller> UNMARSHALLER_MAP = new HashMap<String, Unmarshaller>();
	private Unmarshaller getUnmarshaller() throws JAXBException {
		Unmarshaller unmarshaller = UNMARSHALLER_MAP.get(getSchemaFile());
		if (unmarshaller != null)
			return unmarshaller;

		unmarshaller = JAXB_CONTEXT.createUnmarshaller();
		initUnmarshaller(getSchemaFile(), unmarshaller);
		UNMARSHALLER_MAP.put(getSchemaFile(), unmarshaller);
		return unmarshaller;
	}

	public ResourceLoader() {}


	public void storeResource(S resource, Writer writer)
	throws CICXMLException {
		try {
			JAXBElement<S> element = getJAXBElement(resource);
			DataWriter dataWriter = new DataWriter(writer, "UTF-8");
			dataWriter.setIndentStep("    ");
			getMarshaller().marshal(element, dataWriter);
		} catch (Exception e) {
			throw new CICXMLException(CICXMLErrorCode.XML_MARSHAL_ERROR, e);
		} finally {
			if (writer != null) try {writer.close();} catch (IOException e) {}
		}
	}

	public void storeResource(S resource, File file) throws Exception {
		FileWriter writer = null;
		try {
			writer = new FileWriter(file);
		} catch (IOException e) {
			throw new CICXMLException(CICXMLErrorCode.XML_MARSHAL_ERROR, e);
		}
		storeResource(resource, writer);
	}

	public void storeResource(S resource, OutputStream os) throws CICXMLException {
		storeResource(resource, new OutputStreamWriter(os));
	}

	public S loadResource(String fileName) throws CICXMLException {
		InputStream is;
		try {
			is = new FileInputStream(fileName);
		} catch (IOException e) {
			throw new CICXMLException(CICXMLErrorCode.FILE_OPEN_ERROR, e);
		}

		try {
			S s = loadResource(is);
			return s;
		} finally {
			try {is.close();} catch(IOException e){}
		}
	}

	@SuppressWarnings("unchecked")
	public S loadResource(InputStream is) throws CICXMLException {
		JAXBElement<S> element = null;
		try {
			element = (JAXBElement<S>)getUnmarshaller().unmarshal(is);
		} catch (JAXBException e) {
			throw new CICXMLException(CICXMLErrorCode.XML_UNMARSHAL_ERROR, e);
		}
		return element.getValue();
	}

	abstract protected String getSchemaFile();

	abstract protected JAXBElement<S> getJAXBElement(S resource);

	abstract public S createResource(String name);

}
