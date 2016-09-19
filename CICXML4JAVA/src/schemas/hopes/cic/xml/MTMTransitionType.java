
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for MTMTransitionType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="MTMTransitionType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="condition_list" type="{http://peace.snu.ac.kr/CICXMLSchema}MTMConditionListType" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="dst_mode" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="name" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="src_mode" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "MTMTransitionType", propOrder = {
    "conditionList"
})
public class MTMTransitionType {

    @XmlElement(name = "condition_list", namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected MTMConditionListType conditionList;
    @XmlAttribute(name = "dst_mode", required = true)
    protected String dstMode;
    @XmlAttribute
    protected String name;
    @XmlAttribute(name = "src_mode", required = true)
    protected String srcMode;

    /**
     * Gets the value of the conditionList property.
     * 
     * @return
     *     possible object is
     *     {@link MTMConditionListType }
     *     
     */
    public MTMConditionListType getConditionList() {
        return conditionList;
    }

    /**
     * Sets the value of the conditionList property.
     * 
     * @param value
     *     allowed object is
     *     {@link MTMConditionListType }
     *     
     */
    public void setConditionList(MTMConditionListType value) {
        this.conditionList = value;
    }

    /**
     * Gets the value of the dstMode property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDstMode() {
        return dstMode;
    }

    /**
     * Sets the value of the dstMode property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDstMode(String value) {
        this.dstMode = value;
    }

    /**
     * Gets the value of the name property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the value of the name property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setName(String value) {
        this.name = value;
    }

    /**
     * Gets the value of the srcMode property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSrcMode() {
        return srcMode;
    }

    /**
     * Sets the value of the srcMode property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSrcMode(String value) {
        this.srcMode = value;
    }

}
