
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for MTMType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="MTMType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="mode_list" type="{http://peace.snu.ac.kr/CICXMLSchema}MTMModeListType"/>
 *         &lt;element name="variable_list" type="{http://peace.snu.ac.kr/CICXMLSchema}MTMVariableListType"/>
 *         &lt;element name="transition_list" type="{http://peace.snu.ac.kr/CICXMLSchema}MTMTransitionListType"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "MTMType", propOrder = {
    "modeList",
    "variableList",
    "transitionList"
})
public class MTMType {

    @XmlElement(name = "mode_list", required = true)
    protected MTMModeListType modeList;
    @XmlElement(name = "variable_list", required = true)
    protected MTMVariableListType variableList;
    @XmlElement(name = "transition_list", required = true)
    protected MTMTransitionListType transitionList;

    /**
     * Gets the value of the modeList property.
     * 
     * @return
     *     possible object is
     *     {@link MTMModeListType }
     *     
     */
    public MTMModeListType getModeList() {
        return modeList;
    }

    /**
     * Sets the value of the modeList property.
     * 
     * @param value
     *     allowed object is
     *     {@link MTMModeListType }
     *     
     */
    public void setModeList(MTMModeListType value) {
        this.modeList = value;
    }

    /**
     * Gets the value of the variableList property.
     * 
     * @return
     *     possible object is
     *     {@link MTMVariableListType }
     *     
     */
    public MTMVariableListType getVariableList() {
        return variableList;
    }

    /**
     * Sets the value of the variableList property.
     * 
     * @param value
     *     allowed object is
     *     {@link MTMVariableListType }
     *     
     */
    public void setVariableList(MTMVariableListType value) {
        this.variableList = value;
    }

    /**
     * Gets the value of the transitionList property.
     * 
     * @return
     *     possible object is
     *     {@link MTMTransitionListType }
     *     
     */
    public MTMTransitionListType getTransitionList() {
        return transitionList;
    }

    /**
     * Sets the value of the transitionList property.
     * 
     * @param value
     *     allowed object is
     *     {@link MTMTransitionListType }
     *     
     */
    public void setTransitionList(MTMTransitionListType value) {
        this.transitionList = value;
    }

}
