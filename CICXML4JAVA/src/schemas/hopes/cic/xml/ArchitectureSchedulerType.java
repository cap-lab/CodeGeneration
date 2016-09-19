
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;


/**
 * <p>Java class for ArchitectureSchedulerType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ArchitectureSchedulerType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="RM"/>
 *     &lt;enumeration value="EDF"/>
 *     &lt;enumeration value="RR"/>
 *     &lt;enumeration value="FPFCFS"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum ArchitectureSchedulerType {

    EDF,
    FPFCFS,
    RM,
    RR;

    public String value() {
        return name();
    }

    public static ArchitectureSchedulerType fromValue(String v) {
        return valueOf(v);
    }

}
