
package hopes.cic.xml;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for LibraryConnectionListType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="LibraryConnectionListType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="taskLibraryConnection" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskLibraryConnectionType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="libraryLibraryConnection" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryLibraryConnectionType" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "LibraryConnectionListType", propOrder = {
    "taskLibraryConnection",
    "libraryLibraryConnection"
})
public class LibraryConnectionListType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<TaskLibraryConnectionType> taskLibraryConnection;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected List<LibraryLibraryConnectionType> libraryLibraryConnection;

    /**
     * Gets the value of the taskLibraryConnection property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the taskLibraryConnection property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getTaskLibraryConnection().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link TaskLibraryConnectionType }
     * 
     * 
     */
    public List<TaskLibraryConnectionType> getTaskLibraryConnection() {
        if (taskLibraryConnection == null) {
            taskLibraryConnection = new ArrayList<TaskLibraryConnectionType>();
        }
        return this.taskLibraryConnection;
    }

    /**
     * Gets the value of the libraryLibraryConnection property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the libraryLibraryConnection property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getLibraryLibraryConnection().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link LibraryLibraryConnectionType }
     * 
     * 
     */
    public List<LibraryLibraryConnectionType> getLibraryLibraryConnection() {
        if (libraryLibraryConnection == null) {
            libraryLibraryConnection = new ArrayList<LibraryLibraryConnectionType>();
        }
        return this.libraryLibraryConnection;
    }

}
