
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ArchitectureSchedulerType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ArchitectureSchedulerType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="RM"/&gt;
 *     &lt;enumeration value="EDF"/&gt;
 *     &lt;enumeration value="RR"/&gt;
 *     &lt;enumeration value="FPFCFS"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "ArchitectureSchedulerType")
@XmlEnum
public enum ArchitectureSchedulerType {

    RM,
    EDF,
    RR,
    FPFCFS;

    public String value() {
        return name();
    }

    public static ArchitectureSchedulerType fromValue(String v) {
        return valueOf(v);
    }

}
