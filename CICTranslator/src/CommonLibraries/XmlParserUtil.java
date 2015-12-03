package CommonLibraries;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Vector;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.apache.crimson.tree.XmlDocument;
import org.w3c.dom.CDATASection;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;

public class XmlParserUtil {

    private static final String STENC = "utf-8";
    
    public static ArrayList<Element> getChildElements(Element node, String name)
    {
    	if(node==null) return null;
		ArrayList<Element> childs = new ArrayList<Element>();
		if (node.hasChildNodes()) {
			NodeList list = node.getChildNodes();
			int size = list.getLength();
			for (int i = 0; i < size; i++) {
				Node child = list.item(i);
				if (child.getNodeName().equals(name)) {
					childs.add((Element)child);
				}
			}
		}
		return childs;
    }
    
    public static Vector getChildNodes(Node node, String name) {
    	if(node==null) return null;
		Vector childs = new Vector();
		if (node.hasChildNodes()) {
			NodeList list = node.getChildNodes();
			int size = list.getLength();
			for (int i = 0; i < size; i++) {
				Node child = list.item(i);
				if (child.getNodeName().equals(name)) {
					childs.add(child);
				}
			}
		}
		return childs;
	}
	public static Node findNode(Node node, String name) {
		if (node.getNodeName().equals(name))
			return node;
		if (node.hasChildNodes()) {
			NodeList list = node.getChildNodes();
			int size = list.getLength();
			for (int i = 0; i < size; i++) {
				Node found = findNode(list.item(i), name);
				if (found != null) return found;
			}
		}
		return null;
	}

	public static String getNodeAttribute(Node node, String name) {
		if (node instanceof Element) {
			Element element = (Element)node;
			return element.getAttribute(name);
		}
		return null;
	}
	
	public static String getChildValue(Node node, String name) {
		if (node.hasChildNodes()) {
			NodeList list = node.getChildNodes();
			int size = list.getLength();
			for (int i = 0; i < size; i++) {
				Node child = list.item(i);
				if (child.getNodeName().equals(name)) {
				    try {
				        return child.getFirstChild().getNodeValue();
				    } catch(NullPointerException e) {
				        return "";
				    }
				}
			}
		}
		return null;
	}
	
    public static String getCDATAValue(Node node)
    {
        if(node!=null && node.hasChildNodes()) {
            NodeList list = node.getChildNodes();
            for(int i=0;i<list.getLength();i++) {
                if(list.item(i) instanceof CDATASection)
                {
                    CDATASection section = (CDATASection)list.item(i);
                    return section.getNodeValue();
                }
            }
        }
        return "";
    }
    
	public static Node getChildNode(Node node, String name) {
		if (node.hasChildNodes()) {
			NodeList list = node.getChildNodes();
			int size = list.getLength();
			for (int i = 0; i < size; i++) {
				Node child = list.item(i);
				if (child.getNodeName().equals(name)) {
					return child;
				}
			}
		}
		return null;
	}

    public static String documentToString(Document document)
    {
        StringWriter stWriter = new StringWriter();
        try {
            ((XmlDocument)document).write(stWriter,STENC);
        } catch (Exception e)
        {
            e.printStackTrace();
        }
        return stWriter.toString();
    }
    
    public static Document parse(String xml)
	{ 
		try {
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			DocumentBuilder builder = factory.newDocumentBuilder();
			ByteArrayInputStream stream = new ByteArrayInputStream(xml.getBytes());
			Document doc = builder.parse(stream);
		
			return doc;
		} catch (SAXParseException err) {
            		System.out.println ("** Parsing error"
                		+ ", line " + err.getLineNumber ()
                		+ ", uri " + err.getSystemId ());
            		System.out.println("   " + err.getMessage ());
            		// print stack trace as below
        	} catch (SAXException e) {
            		Exception   x = e.getException (); 
			((x == null) ? e : x).printStackTrace ();
        	} catch (Throwable t) {
            		t.printStackTrace ();
        	}

		return null;
	}
    
    public static Document parse(File xml)
    {
    	{ 
    		try {
    			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
    			DocumentBuilder builder = factory.newDocumentBuilder();
    			Document doc = builder.parse(xml);
    		
    			return doc;
    		} catch (SAXParseException err) {
                		System.out.println ("** Parsing error"
                    		+ ", line " + err.getLineNumber ()
                    		+ ", uri " + err.getSystemId ());
                		System.out.println("   " + err.getMessage ());
                		// print stack trace as below
            	} catch (SAXException e) {
                		Exception   x = e.getException (); 
    			((x == null) ? e : x).printStackTrace ();
            	} catch (Throwable t) {
                		t.printStackTrace ();
            	}

    		return null;
    	}        
    }
    
    public static void addValuedElement(Document doc, Element element, String name, String value)
    {
        Element subElem = doc.createElement(name);
        Text text = doc.createTextNode(value);
        subElem.appendChild(text);
        element.appendChild(subElem);
    }
}